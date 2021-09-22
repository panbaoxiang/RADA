hype={RandomSample[{300,500,1000}][[1]],RandomSample[{5,7,10}][[1]],RandomSample[Range[1,3]][[1]]};
days={1,0}
dim={1,26,48};
dim2={3*(Total[days]+1),36,56};
nP4Obser=nP4Obser[[1+days[[1]];;Length[nP4Obser]-days[[2]]]];
ndynamics4Obser=Map[Flatten[#,1]&,Transpose[Table[ndynamics4Obser[[1+k;;Length[nP4Obser]+k]],{k,0,Total[days]}]]];
nP4GCM=nP4GCM[[1+days[[1]];;Length[nP4GCM]-days[[2]]]];
ndynamics4GCM=Map[Flatten[#,1]&,Transpose[Table[ndynamics4GCM[[1+k;;Length[nP4GCM]+k]],{k,0,Total[days]}]]];
mse[a_,b_]:=Mean[(a-b)^2]
length=14610;
vlength=3652;
tlength=Length[nP4GCM];
test=Table[<|"P_GCM"->nP4GCM[[i]],
             "D_GCM"->ndynamics4GCM[[i]],
             "P_Obser"->nP4Obser[[i]],
             "D_Obser"->ndynamics4Obser[[i]]
             |>,{i,length+vlength+1,Length[nP4GCM]}];

seq=Flatten[{Select[Flatten[Table[Range[(i-1)*365+1,i*365],{i,Select[Range[1,60],Mod[#,10]<=7&]}]],#<=length+vlength&],
	    Select[Flatten[Table[Range[(i-1)*365+1,i*365],{i,Select[Range[1,60],Mod[#,10]>7&]}]],#<=length+vlength&]}];
{nP4GCM,ndynamics4GCM,nP4Obser,ndynamics4Obser}=Map[#[[seq]]&,{nP4GCM,ndynamics4GCM,nP4Obser,ndynamics4Obser}];

trainObser=Table[ndynamics4Obser[[i]]->nP4Obser[[i]],{i,length}];
validationObser=Table[ndynamics4Obser[[i]]->nP4Obser[[i]],{i,length,length+vlength}];

downscalingObser=Import["/g/g92/pan11/Backup_CycleGAN/Previous/Downscaling_Obser.mx"];

trainedObser=NetTrain[downscalingObser,trainObser,
	ValidationSet->validationObser,
	TargetDevice->{"GPU",All},
	Method->{"ADAM","L2Regularization"->10^-3,"LearningRate"->10^-4},
	MaxTrainingRounds->500,
	BatchSize->32];
trainedObser=NetTrain[trainedObser,trainObser,
	ValidationSet->validationObser,
	TargetDevice->{"GPU",All},
	Method->{"ADAM","L2Regularization"->10^-3,"LearningRate"->10^-5},
	MaxTrainingRounds->500,
	BatchSize->32];
trainedObser=NetTrain[trainedObser,trainObser,
	ValidationSet->validationObser,
	TargetDevice->{"GPU",All},
	Method->{"ADAM","L2Regularization"->10^-3,"LearningRate"->10^-6},
	MaxTrainingRounds->500,
	BatchSize->32];

simu=Table[trainedObser[ndynamics4Obser[[i]],TargetDevice->"GPU"],{i,Length[ndynamics4Obser]}];
obser=nP4Obser;

corr=Table[If[And[Variance[simu[[;;,1,i,j]]]>0,Variance[obser[[;;,1,i,j]]]>0],
	Correlation[simu[[;;,1,i,j]],obser[[;;,1,i,j]]],-2],{i,26},{j,48}];
Mean[Select[Flatten[corr],Positive]]

amse=Table[If[And[Variance[simu[[;;,1,i,j]]]>0,Variance[obser[[;;,1,i,j]]]>0],
        mse[simu[[;;,1,i,j]],obser[[;;,1,i,j]]],0],{i,26},{j,48}];
Mean[Flatten[amse]]
Export["/g/g92/pan11/Backup_CycleGAN/Downscaling_Obser.mx",
<|"net"->trainedObser,
  "mse"->Mean[Flatten[amse]]|>]


trainGCM=Table[ndynamics4GCM[[i]]->nP4GCM[[i]],{i,length}];
validationGCM=Table[ndynamics4GCM[[i]]->nP4GCM[[i]],{i,length,length+vlength}];

downscalingGCM=Import["/g/g92/pan11/Backup_CycleGAN/Previous/Downscaling_GCM.mx"];
trainedGCM=NetTrain[downscalingGCM,trainGCM,
	ValidationSet->validationGCM,
	TargetDevice->{"GPU",All},
	Method->{"ADAM","L2Regularization"->10^-3,"LearningRate"->10^-4},
	MaxTrainingRounds->500,
	BatchSize->32];
trainedGCM=NetTrain[trainedGCM,trainGCM,
	ValidationSet->validationGCM,
	TargetDevice->{"GPU",All},
	Method->{"ADAM","L2Regularization"->10^-3,"LearningRate"->10^-5},
	MaxTrainingRounds->500,
	BatchSize->32];
trainedGCM=NetTrain[trainedGCM,trainGCM,
	ValidationSet->validationGCM,
	TargetDevice->{"GPU",All},
	Method->{"ADAM","L2Regularization"->10^-3,"LearningRate"->10^-6},
	MaxTrainingRounds->500,
	BatchSize->32];

simu=Table[trainedGCM[ndynamics4GCM[[i]],TargetDevice->"GPU"],{i,Length[ndynamics4GCM]}];
obser=nP4GCM;

corr=Table[If[And[Variance[simu[[;;,1,i,j]]]>0,Variance[obser[[;;,1,i,j]]]>0],
        Correlation[simu[[;;,1,i,j]],obser[[;;,1,i,j]]],-2],{i,26},{j,48}];
Mean[Select[Flatten[corr],Positive]]

amse=Table[If[And[Variance[simu[[;;,1,i,j]]]>0,Variance[obser[[;;,1,i,j]]]>0],
        mse[simu[[;;,1,i,j]],obser[[;;,1,i,j]]],0],{i,26},{j,48}];
Mean[Flatten[amse]]

Export["/g/g92/pan11/Backup_CycleGAN/Downscaling_GCM.mx",
<|"net"->trainedGCM,
  "mse"->Mean[Flatten[amse]]|>]
