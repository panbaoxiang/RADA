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
vposition=Position[Flatten[validMatrix],1][[;;,1]];

test=Table[<|"P_GCM"->nP4GCM[[i]],
             "P_Obser"->nP4Obser[[i]]
             |>,{i,length+vlength+1,Length[nP4GCM]}];
gsimu=test[[;;,1]];
gobser=test[[;;,2]];

seq=Flatten[{Select[Flatten[Table[Range[(i-1)*365+1,i*365],{i,Select[Range[1,60],Mod[#,10]<=7&]}]],#<=length+vlength&],
            Select[Flatten[Table[Range[(i-1)*365+1,i*365],{i,Select[Range[1,60],Mod[#,10]>7&]}]],#<=length+vlength&]}];
{nP4GCM,ndynamics4GCM,nP4Obser,ndynamics4Obser}=Map[#[[seq]]&,{nP4GCM,ndynamics4GCM,nP4Obser,ndynamics4Obser}];

training=Table[<|"P_GCM"->nP4GCM[[i]],
             "P_Obser"->nP4Obser[[i]]
             |>,{i,length}];

validation=Table[<|"P_GCM"->nP4GCM[[i]],
             "P_Obser"->nP4Obser[[i]]
             |>,{i,length+1,Length[nP4GCM]}];

position=Position[validMatrix,1];
n=Length[position];

training=Table[<|"P_GCM"->Flatten[training[[i,1]]][[Position[Flatten[validMatrix],1][[;;,1]]]],
		 "P_Obser"->Flatten[training[[i,2]]][[Position[Flatten[validMatrix],1][[;;,1]]]]|>,{i,Length[training]}];
validation=Table[<|"P_GCM"->Flatten[validation[[i,1]]][[Position[Flatten[validMatrix],1][[;;,1]]]],
		 "P_Obser"->Flatten[validation[[i,2]]][[Position[Flatten[validMatrix],1][[;;,1]]]]|>,{i,Length[validation]}];
test=Table[<|"P_GCM"->Flatten[test[[i,1]]][[Position[Flatten[validMatrix],1][[;;,1]]]],
		 "P_Obser"->Flatten[test[[i,2]]][[Position[Flatten[validMatrix],1][[;;,1]]]]|>,{i,Length[test]}];

total=23;
SetDirectory["/usr/workspace/pan11/CycleGAN/MBCn"];
energy=Table[Import["Step_"<>ToString[i]<>".mx"]["energy"],{i,total}];
simu=test[[;;,1]];
obser=test[[;;,2]];
seq=RandomSample[Range[Length[training]]][[1;;Length[validation]]];


MBCn=Table[Block[{matrix,transform,s1,s2,correction},
matrix=Import["Step_"<>ToString[i]<>".mx"]["matrix"];
transform=Import["Step_"<>ToString[i]<>".mx"]["transform"];
transform=Table[Sort[DeleteDuplicates[transform[[dim,;;,1;;2]]],#1[[1,2]]<=#2[[1,2]]&],{dim,Length[transform]}];
s1=simu.matrix;
Print[{i,Correlation[Mean[obser],Mean[simu]]}];
s2=ParallelTable[Block[{raw=s1[[;;,dim]],po},
 result=Map[Block[{ele=#},If[ele<=transform[[dim,1,1,1]],transform[[dim,2,1,1]],
       If[ele>=transform[[dim,-1,1,2]],transform[[dim,-1,2,2]],
        Block[{po=transform[[dim,Position[transform[[dim,;;,1,1]],_?(#<=ele&)][[-1,1]]]]},
	  (ele-po[[1,1]])*(po[[1,2]]-po[[1,1]])+po[[2,1]]]]]]&,raw];
 result],{dim,Dimensions[s1][[2]]}];
correction=Transpose[s2].Inverse[matrix];
(*correction=correction/. x_ /; x<0.->0;*)
Set[simu,correction];
simu],{i,total}];

final=Block[{tempt=Table[0,{i,Dimensions[MBCn][[2]]},{j,Dimensions[validMatrix][[1]]},{k,Dimensions[validMatrix][[2]]}],np},
 Table[Set[tempt[[;;,position[[i,1]],position[[i,2]]]],MBCn[[-1,;;,i]]],{i,Length[position]}];
 np=Position[tempt,_?(#<0&)];
 Map[Set[tempt[[#[[1]],#[[2]],#[[3]]]],0]&,np];
 tempt];

Export["/usr/workspace/pan11/CycleGAN/MBCn/MBCn_Result.mx",{NumericArray[Exp[final]-1.,"Real32"],NumericArray[Exp[gsimu[[;;,1]]]-1.,"Real
32"],NumericArray[Exp[gobser[[;;,1]]]-1.,"Real32"]}]



total=1;
SetDirectory["/usr/workspace/pan11/CycleGAN/MBCn"];
energy=Table[Import["Step_"<>ToString[i]<>".mx"]["energy"],{i,total}];
simu=test[[;;,1]];
obser=test[[;;,2]];
seq=RandomSample[Range[Length[training]]][[1;;Length[validation]]];


MBCn=Table[Block[{matrix,transform,s1,s2,correction},
matrix=Import["Step_"<>ToString[i]<>".mx"]["matrix"];
transform=Import["Step_"<>ToString[i]<>".mx"]["transform"];
transform=Table[Sort[DeleteDuplicates[transform[[dim,;;,1;;2]]],#1[[1,2]]<=#2[[1,2]]&],{dim,Length[transform]}];
s1=simu.matrix;
Print[{i,Correlation[Mean[obser],Mean[simu]]}];
s2=ParallelTable[Block[{raw=s1[[;;,dim]],po},
 result=Map[Block[{ele=#},If[ele<=transform[[dim,1,1,1]],transform[[dim,2,1,1]],
       If[ele>=transform[[dim,-1,1,2]],transform[[dim,-1,2,2]],
        Block[{po=transform[[dim,Position[transform[[dim,;;,1,1]],_?(#<=ele&)][[-1,1]]]]},
          (ele-po[[1,1]])*(po[[1,2]]-po[[1,1]])+po[[2,1]]]]]]&,raw];
 result],{dim,Dimensions[s1][[2]]}];
correction=Transpose[s2].Inverse[matrix];
(*correction=correction/. x_ /; x<0.->0;*)
Set[simu,correction];
simu],{i,total}];

final=Block[{tempt=Table[0,{i,Dimensions[MBCn][[2]]},{j,Dimensions[validMatrix][[1]]},{k,Dimensions[validMatrix][[2]]}],np},
 Table[Set[tempt[[;;,position[[i,1]],position[[i,2]]]],MBCn[[-1,;;,i]]],{i,Length[position]}];
 np=Position[tempt,_?(#<0&)];
 Map[Set[tempt[[#[[1]],#[[2]],#[[3]]]],0]&,np];
 tempt];

Export["/usr/workspace/pan11/CycleGAN/MBCn/MBCn_Result_EarlyStopping.mx",{NumericArray[Exp[final]-1.,"Real32"],NumericArray[Exp[gsimu[[;;,1]]]-1.,"Real32"],NumericArray[Exp[gobser[[;;,1]]]-1.,"Real32"]}]
