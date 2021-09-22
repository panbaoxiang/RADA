SetDirectory["/g/g92/pan11/Trained"];
models=Map[Import,FileNames["Cycle*mx"]];

Import["/g/g92/pan11/CycleGAN/2020_11_13_CycleGAN_Data.m"];
days={1,0}
dim={1,26,48};
dim2={3*(Total[days]+1),36,56};
nP4Obser=nP4Obser[[1+days[[1]];;Length[nP4Obser]-days[[2]]]];
ndynamics4Obser=Map[Flatten[#,1]&,Transpose[Table[ndynamics4Obser[[1+k;;Length[nP4Obser]+k]],{k,0,Total[days]}]]];
nP4GCM=nP4GCM[[1+days[[1]];;Length[nP4GCM]-days[[2]]]];
ndynamics4GCM=Map[Flatten[#,1]&,Transpose[Table[ndynamics4GCM[[1+k;;Length[nP4GCM]+k]],{k,0,Total[days]}]]];
length=14610;
vlength=3652;
test=Table[<|"P_GCM"->nP4GCM[[i]],
             "D_GCM"->ndynamics4GCM[[i]],
             "P_Obser"->nP4Obser[[i]],
             "D_Obser"->ndynamics4Obser[[i]]
             |>,{i,length+vlength+1,Length[nP4GCM]}];

DADT=Block[{ms},
	ms=Map[#[["Generator_GCM->Obser"]]&,models];
	Table[Block[{tempt=ms[[i]]},Print[i];
	Map[tempt[<|"P"->#[["P_GCM"]],"z"->#[["D_GCM"]]|>,TargetDevice->"GPU"]&,test]],
	{i,Length[models]}]][[;;,;;,1]];
GCM=test[[;;,"P_GCM"]][[;;,1]];
OBSER=test[[;;,"P_Obser"]][[;;,1]];

{DADT,GCM,OBSER}=Map[Exp[#]-1&,{DADT,GCM,OBSER}];
Export["/g/g92/pan11/result.mx",Map[NumericArray[#,"Real32"]&,{DADT,GCM,OBSER}]];

downscalingObser=Block[{models},
	SetDirectory["/g/g92/pan11/Trained"];
	models=Map[Import[#][["R_Downscaling_Obser"]]&,FileNames["Cycle*mx"]];
	Table[Block[{tempt=models[[i]]},Print[i];
	Map[{Exp[tempt[#[["D_GCM"]],TargetDevice->"GPU"]]-1.,
	     Exp[tempt[#[["D_Obser"]],TargetDevice->"GPU"]]-1.}&,test]],
	{i,Length[models]}]];
Export["/g/g92/pan11/DownscalingObser.mx",NumericArray[downscalingObser,"Real32"]];

downscalingGCM=Block[{models},
	SetDirectory["/g/g92/pan11/Trained"];
	models=Map[Import[#][["R_Downscaling_GCM"]]&,FileNames["Cycle*mx"]];
	Table[Block[{tempt=models[[i]]},Print[i];
	Map[{Exp[tempt[#[["D_GCM"]],TargetDevice->"GPU"]]-1.,
	     Exp[tempt[#[["D_Obser"]],TargetDevice->"GPU"]]-1.}&,test]],
	{i,Length[models]}]];
Export["/g/g92/pan11/DownscalingGCM.mx",NumericArray[downscalingGCM,"Real32"]];
