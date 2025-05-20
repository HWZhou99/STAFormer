
        
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from torchvision import models
from models.backbone.resnest import resnest50

from models.key_model.gempool import GeneralizedMeanPoolongP

from models.key_model.SpatialAttention2d import SpatialAttention2d
from models.key_model.vit import ViT
from models.key_model.patch import PatchEmbed


''' 
python main.py   --root /home/cv-mot/DLF/DataSet/MARS-v160809  --dataset mars --arch make_model --gpu 0,1 --save_dir test

python main_yuesu.py   --root /home/cv-mot/DLF/DataSet/iLIDS-VID  --dataset ilidsvid --arch make_model --gpu 0,1 --save_dir log_make_model_spa_kp_ilidsvid


python main.py --arch make_model --dataset mars --root /home/cv-mot/DLF/DataSet/MARS-v160809 --gpu_devices 0,1 --save_dir /home/cv-mot/zhw/STAFormer/log_make_model_spa_kp --evaluate --all_frames --resume /home/cv-mot/zhw/STAFormer/log_make_model_spa_kp/checkpoint_ep200.pth.tar

python main_yuesu.py --arch make_model --dataset ilidsvid --root /home/cv-mot/DLF/DataSet/iLIDS-VID --gpu_devices 0,1 --save_dir log_make_model_spa_kp_ilidsvid --evaluate --all_frames --resume  /home/cv-mot/zhw/ResNeSt_mixer/log_make_model_spa_kp_ilidsvid/checkpoint_ep150.pth.tar

'''
def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight,std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias,0.0)

def weight_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight,a=0,mode='fan_out')
        nn.init.constant_(m.bias,0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight,a=0,mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias,0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight,1.0)
            nn.init.constant_(m.bias,0.0)

class CustomRes(nn.Module):
    def __init__(self,original_model,layers_to_extract):
        super(CustomRes,self).__init__()
        self.features=nn.Sequential(
            *list(original_model.children())[:-2]
        )
        self.layers_to_extract=layers_to_extract
    def forward(self, x):
        outputs = []
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in self.layers_to_extract:
                outputs.append(x)
        return outputs
    
class make_model(nn.Module):
    def __init__(self,num_classes,seq_len) :
        super().__init__()
        self.dim=768
        self.num_classes=num_classes
        self.seq_len=seq_len#frame_len
        self.depth=3#number of encoder layer
        self.heads=12
        self.dim_head=self.dim//self.heads#dim=heads*dim_head
        self.mlp_dim=3072
        self.loc_dim=1024
        self.glo_dim=2048
        
        self.use_loc_spa=True
    
        resnest=resnest50(pretrained=True)
        
        #Select the layer from which you want to extract the results 6 for stage3 and 7 for stage4.
        layers_to_extract=['6','7']
        
        self.custom_resnest=CustomRes(resnest,layers_to_extract)
        #
        self.lfa=SpatialAttention2d(in_c=1024)
     
        self.gem=GeneralizedMeanPoolongP()
        
        # Local branching constraints,
        self.bottleneck_loc = nn.BatchNorm1d(self.loc_dim)
        self.classifier_loc = nn.Linear(self.loc_dim, self.num_classes)
        self.bottleneck_loc.apply(weight_init_kaiming)
        self.classifier_loc.apply(weight_init_classifier)
        #Global Branching Constraints
        self.bottleneck_glo = nn.BatchNorm1d(self.glo_dim)
        self.classifier_glo = nn.Linear(self.glo_dim, self.num_classes)
        self.bottleneck_glo.apply(weight_init_kaiming)
        self.classifier_glo.apply(weight_init_classifier)
        #Divide patch 
        self.patch_embed_loc=PatchEmbed(img_size=(1,1),patch_size=1,stride_size=1,in_chans=self.loc_dim, embed_dim=self.dim)
        self.patch_embed_glo=PatchEmbed(img_size=(1,1),patch_size=1,stride_size=1,in_chans= self.glo_dim,embed_dim=self.dim)
        # Attention-based Feature Aggregation(AFA)
        self.mix_vit=ViT(
            num_patches=9,
            dim=self.dim,
            depth=3,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
            dim_head=self.dim_head,
            pool='cls',
            dropout = 0.1
        )
     
        self.vit=ViT(
            num_patches=self.seq_len,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
            dim_head=self.dim_head,
            pool='mean',
            dropout = 0.1
        )
       
        '''classifical'''
        self.bn=nn.BatchNorm1d(self.dim)
        self.bn.apply(weight_init_kaiming)
        self.classifier=nn.Linear(self.dim,self.num_classes)
        self.classifier.apply(weight_init_classifier)
    
        
        self.projection=nn.Conv1d(self.dim,self.dim,(1,),(1,))
        self.projection.apply(weight_init_kaiming)
    def forward(self,x):
        b,c,t,h,w=x.size()
        x=x.permute(0,2,1,3,4).contiguous()
        x=x.view(b*t,c,h,w)
        
        cls_score_list_base = []
        bn_feat_list_base = []
        
        loc_feats,glo_feats=self.custom_resnest(x)

        glo_feat=self.gem(glo_feats)
     
        #######################Local branching constraints############################################################# 
        cnn_feat_loc = loc_feats.mean(-1).mean(-1)#.reshape(B, T, C).mean(1)
        bn_feature_loc = self.bottleneck_loc(cnn_feat_loc)
        cls_score_loc = self.classifier_loc(bn_feature_loc)
        cls_score_list_base.append(cls_score_loc)
        bn_feat_list_base.append(bn_feature_loc)
           
        ###############################Global Branching Constraints ####################################################### 
        # cnn_feat_glo = glo_feats.mean(-1).mean(-1)#.reshape(B, T, C).mean(1)
        bn_feature_glo = self.bottleneck_glo(glo_feat.squeeze())
        cls_score_glo = self.classifier_glo(bn_feature_glo)
        cls_score_list_base.append(cls_score_glo)
        bn_feat_list_base.append(bn_feature_glo)
        ###############################融合约束 #######################################################     
        
        if self.use_loc_spa==True:
            #Local feature map
            loc_feat_map=self.lfa(loc_feats)
            att = loc_feat_map.expand_as(loc_feats)
            loc_feat=att*loc_feats
        else:
            loc_feat=loc_feats
            
        #Extract key points using maximum pooling
        key_point=F.max_pool2d(loc_feat,(4,4))#bt,1024,4,2
        
        #Divide the patch
        loc = self.patch_embed_loc(key_point)
        
        glo=self.patch_embed_glo(glo_feat)
        loc_glo_mix=torch.cat((loc,glo),dim=1)
        
        fusing=self.mix_vit(loc_glo_mix)
        fusing=fusing[:,0]


        fusing_x=fusing.reshape(b,t,-1)
        x=self.vit(fusing_x)
      
        x = x.transpose(1, 2)#torch.Size([b, 1024, T])
        if not self.training:
            f=self.bn(x)
            return f
        v=x.mean(-1)
        f=self.bn(v)
        y=self.classifier(f)
        
        i= self.bn(x)
        i= self.projection(i)
        return i,y,f,cls_score_list_base,bn_feat_list_base

                 
        
        