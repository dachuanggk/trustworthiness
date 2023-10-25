clear,clc

delta=0:0.01:1;
len=length(delta);
Q=zeros(len,3);


 mu(:,:,1)=[0.49 0.50 0.46 0.42
     0.57 0.43 0.35 0.50
     0.44 0.58 0.46 0.55];

eta(:,:,1)=[0.21  0.21  0.31 0.23
      0.16  0.22  0.23 0.27 
      0.20  0.08  0.17 0.22];
      
nu(:,:,1)=[0.18 0.28  0.22  0.33
     0.13 0.20  0.35  0.19
     0.21 0.17  0.26  0.21];

mu(:,:,2)=[0.36 0.45  0.37 0.52          
     0.49 0.38  0.46 0.48
     0.53 0.49  0.51 0.49 ];

eta(:,:,2)=[0.23  0.18  0.22 0.16       
      0.18  0.07  0.14 0.11
      0.15  0.11  0.02 0.10];
  
nu(:,:,2)=[0.12 0.20  0.22  0.11
     0.23 0.18  0.36  0.09
     0.20 0.16  0.25  0.13];
 
 mu(:,:,3)=[0.47 0.55 0.46 0.53
     0.48 0.43 0.35 0.51
     0.44 0.59 0.48 0.56];

eta(:,:,3)=[0.19  0.15  0.20 0.30
      0.15  0.21  0.22 0.26 
      0.18  0.07  0.16 0.21];
      
nu(:,:,3)=[0.21 0.21  0.32  0.12
     0.23 0.19  0.36  0.19
     0.20 0.16  0.25  0.10];

mu(:,:,4)=[0.55 0.51  0.51 0.50  
     0.37 0.46  0.38 0.53
     0.50 0.39  0.57 0.49];

eta(:,:,4)=[0.22  0.17  0.21 0.15
      0.22 0.17  0.35  0.08     
      0.17  0.06  0.13 0.10];
  
nu(:,:,4)=[0.11 0.19  0.21  0.10
     0.14  0.11  0.12 0.11 
     0.19 0.15  0.23  0.12];

disp('第1步，输出4个矩阵');

[m,n]=size(mu(:,:,1));
t=4;

for k=1:t
    temp=[reshape(mu(:,:,k)',1,[]); reshape(eta(:,:,k)',1,[]); reshape(nu(:,:,k)',1,[])];
   eval(['X',num2str(k),'=  sprintf(''(%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)\n'', temp)'])
end
clear X1 X2 X3 X4
%%%% 第2步，求加权决策矩阵
disp('第2步，weights=[0.3,0.2,0.3,0.2]的加权决策')

weights=[0.3,0.2,0.3,0.2];
weight=ones(m,1)*weights;

for k=1:t  
    tau(:,:,k)=1-(1-mu(:,:,k)).^weight;  
    varsigma(:,:,k)=eta(:,:,k).^weight;
    upsilon(:,:,k)=(nu(:,:,k)+eta(:,:,k)).^weight-eta(:,:,k).^weight;
 %   upsilon(:,:,k)=nu(:,:,k).^weight;
    
   tmp=[reshape(tau(:,:,k)',1,[]); reshape(varsigma(:,:,k)',1,[]); reshape(upsilon(:,:,k)',1,[])];
   eval(['Y',num2str(k),'=  sprintf(''(%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)\n'', tmp)']);
end

%%%% 第3步，求arithmetic mean
tau_mean=1; varsigma_mean=1; upsilon_mean=1; up_mean=1;
for k=1:t     
    tau_mean=tau_mean.*(1-tau(:,:,k)).^(1/4);  
    varsigma_mean=varsigma_mean.*varsigma(:,:,k).^(1/4); 
    up_mean= up_mean.*(upsilon(:,:,k)+varsigma(:,:,k)).^(1/4);   %这里，？

%      tau_mean=prod((1-tau(:,:,k)).^(1/4));  
%      varsigma_mean=prod(varsigma(:,:,k).^(1/4)); 
%     upsilon_mean=prod((upsilon(:,:,k)+varsigma(:,:,k)).^(1/4))-varsigma_mean;   %这里，？
end
tau_mean=1-tau_mean;
varsigma_mean;
up_mean;
upsilon_mean=up_mean-varsigma_mean;

disp('Arithmetic mean')

tmp=[reshape(tau_mean',1,[]); reshape(varsigma_mean',1,[]); reshape(upsilon_mean',1,[])];
    sprintf('(%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)\n', tmp)

% sprintf('(%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)\n', [tau_mean;varsigma_mean;upsilon_mean])

for k=1:t 
pi(:,:,k)=1-tau(:,:,k)-varsigma(:,:,k)-upsilon(:,:,k);
E(k)=-sum(sum(tau(:,:,k).*log(tau(:,:,k))+varsigma(:,:,k).*log(varsigma(:,:,k))+upsilon(:,:,k).*log(upsilon(:,:,k))+pi(:,:,k).*log(pi(:,:,k))))/((m*n)*log(m*n)); 
end  %% 留下3个数和大于1的例子
disp('Entropy of Yk')
E

pi_me=tau_mean+varsigma_mean+upsilon_mean;

pi_mean=1-pi_me;
E_mean=-sum(sum(tau_mean.*log(tau_mean)+varsigma_mean.*log(varsigma_mean)+upsilon_mean.*log(upsilon_mean)+pi_mean.*log(pi_mean)))/((m*n)*log(m*n))

dist=abs(E-E_mean);
ma=max(dist);
RC=ma./(ma+dist)

lambda=RC/sum(RC)

%%%% 第4步，将标准阵加决策者权
for k=1:t
    
    kappa(:,:,k)=(1-tau(:,:,k)).^lambda(k);
    iota(:,:,k)=varsigma(:,:,k).^lambda(k);
    varphi(:,:,k)=(varsigma(:,:,k)+upsilon(:,:,k)).^lambda(k)-varsigma(:,:,k).^lambda(k);
    
    kappa(:,:,k)=1-kappa(:,:,k);

   tmp=[reshape(kappa(:,:,k)',1,[]); reshape(iota(:,:,k)',1,[]); reshape(varphi(:,:,k)',1,[])];
   eval(['H',num2str(k),'=  sprintf(''(%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)\n'', tmp)']);
end

   %%%% 转换成方案矩阵

disp('第10步，群决策矩阵')
 kappai=permute(kappa,[3,2,1]);
 iotai=permute(iota,[3,2,1]);
 varphii=permute(varphi,[3,2,1]);
 
for i=1:3  
     temp=[reshape(kappai(:,:,i)',1,[]); reshape(iotai(:,:,i)',1,[]); reshape(varphii(:,:,i)',1,[])];
    eval(['Hi',num2str(i),'=  sprintf(''(%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)\n'', temp)']);
end
   

for i=1:3      
    kappai_pos=max(kappai(:,:,i),[],3); %%%正理想解
    kappai_neg=min(kappai(:,:,i),[],3);%%%负理想解
end
kappai_pos;
kappai_neg;

for i=1:3   
    iotai_pos=min(iotai,[],3); %%%正理想解
    iotai_neg=max(iotai,[],3); %%%负理想解
    
end
iotai_pos;
iotai_neg;


for i=1:3     
    varphii_pos=min(varphii,[],3);%%%正理想解   
    varphii_neg=max(varphii,[],3); %%%负理想解
end
varphii_pos;
varphii_neg;


pos=[];
neg=[];
com=[];
pos=[pos, kappai_pos(:,1), iotai_pos(:,1), varphii_pos(:,1),kappai_pos(:,2), iotai_pos(:,2), varphii_pos(:,2),kappai_pos(:,3), iotai_pos(:,3), varphii_pos(:,3),kappai_pos(:,4), iotai_pos(:,4), varphii_pos(:,4)];
neg=[neg, kappai_neg(:,1), iotai_neg(:,1), varphii_neg(:,1),kappai_neg(:,2), iotai_neg(:,2), varphii_neg(:,2),kappai_neg(:,3), iotai_neg(:,3), varphii_neg(:,3),kappai_neg(:,4), iotai_neg(:,4), varphii_neg(:,4)];
com=[com, varphii_pos(:,1), iotai_pos(:,1), kappai_pos(:,1),varphii_pos(:,2), iotai_pos(:,2), kappai_pos(:,2),varphii_pos(:,3), iotai_pos(:,3), kappai_pos(:,3),varphii_pos(:,4), iotai_pos(:,4), kappai_pos(:,4)];



disp('第3步，理想决策')
%%% 输出正负理想解 
disp('正理想决策H_{+}为：')
sprintf('(%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)\n', pos')
disp('负理想决策H_{-}为：')
 sprintf('(%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)\n', neg')
disp('余理想决策H_{c}为：')
sprintf('(%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)\n', com')



%%%% 8个内积

for i=1:3 
    HH_pos(i)=sum(sum(kappai(:,:,i).*kappai_pos+iotai(:,:,i).*iotai_pos+varphii(:,:,i).*varphii_pos ...
     +(1-kappai(:,:,i)-iotai(:,:,i)-varphii(:,:,i)).*(1-kappai_pos-iotai_pos-varphii_pos)));
 
     H_mod_squ(i)=sum(sum(kappai(:,:,i).^2+iotai(:,:,i).^2+varphii(:,:,i).^2 ...
     +(1-kappai(:,:,i)-iotai(:,:,i)-varphii(:,:,i)).^2));
 
    HH_neg(i)=sum(sum(kappai(:,:,i).*kappai_neg+iotai(:,:,i).*iotai_neg+varphii(:,:,i).*varphii_neg ...
     +(1-kappai(:,:,i)-iotai(:,:,i)-varphii(:,:,i)).*(1-kappai_neg-iotai_neg-varphii_neg)));
 
     HH_com(i)=sum(sum(kappai(:,:,i).*varphii_pos+iotai(:,:,i).*iotai_pos+varphii(:,:,i).*kappai_pos ...
     +(1-kappai(:,:,i)-iotai(:,:,i)-varphii(:,:,i)).*(1-kappai_pos-iotai_pos-varphii_pos)));
end

disp('Hi与H_{+}内积为：')
HH_pos

 disp('Hi与H_{-}内积为：')

 HH_neg
 
disp('Hi与H_{c}内积为：')
HH_com

disp('H_{i}^{2}为：')
H_mod_squ

disp('H_{+}^{2}为：')
H_mod_pos=sum(sum(kappai_pos.^2+iotai_pos.^2+varphii_pos.^2+(1-kappai_pos-iotai_pos-varphii_pos).^2))

disp('H_{-}^{2}为：')
 H_mod_neg=sum(sum(kappai_neg.^2+iotai_neg.^2+varphii_neg.^2+(1-kappai_neg-iotai_neg-varphii_neg).^2))
 

%%%% 第5步，求4个射影    
    
    for i=1:i
        NPH_p(i)=min(HH_pos(i)-H_mod_squ(i)+H_mod_pos,HH_pos(i)+H_mod_squ(i)-H_mod_pos)./max(HH_pos(i)-H_mod_squ(i)+H_mod_pos,HH_pos(i)+H_mod_squ(i)-H_mod_pos);
        NPH_n(i)=min(HH_neg(i)-H_mod_squ(i)+H_mod_neg,HH_neg(i)+H_mod_squ(i)-H_mod_neg)./max(HH_neg(i)-H_mod_squ(i)+H_mod_neg,HH_neg(i)+H_mod_squ(i)-H_mod_neg);
        NPH_c(i)=min(HH_com(i)-H_mod_squ(i)+H_mod_pos,HH_com(i)+H_mod_squ(i)-H_mod_pos)./max(HH_com(i)-H_mod_squ(i)+H_mod_pos,HH_com(i)+H_mod_squ(i)-H_mod_pos);
    end
    
    disp('Hi到H_{+}的标准化投影为：')
  
  NPH_p 
  
 disp('Hi到H_{-}的标准化投影为：')
  
  NPH_n  
  
  disp('Hi到H_{c}的标准化投影为：')
  
  NPH_c 
  
 
       
     GU=NPH_p
  
     GR=(NPH_n+NPH_c)/2 
      
       GU_pos=max(GU)
       GU_neg=min(GU)
       
       GR_neg=max(GR)
       GR_pos=min(GR)
       
       NGU=(GU-GU_neg)/(GU_pos-GU_neg)
       
       NGR=(GR_neg-GR)/(GR_neg-GR_pos)
      
%        Q=0.9*NGU+0.1*NGR
      
       for row=1:len
           
            Q=delta(row)*NGU+(1-delta(row))*NGR;
            
            Q1(row,:)=Q;
            
%       RC=(2/3-delta(row))*RC2 +delta(row)*RC3+(RC1)/3;

end

% 绘图
% figure,set(gcf,'outerposition',get(0,'screensize')); 
plot(delta,Q1(:,1),'b'),hold on,
plot(delta,Q1(:,2),'k-.'),plot(delta,Q1(:,3),'g--');
%,plot(delta,Q(:,4),'k-.');
legend('A_{1}','A_{2}','A_{3}','Location','Southeast');
% title('\delta');
xlabel('\lambda');
ylabel('Ranking of alternatives');


      


