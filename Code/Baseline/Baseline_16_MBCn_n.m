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


gcm=training[[;;,1]];
obser=training[[;;,2]];
tobser=test[[;;,2]];
vobser=validation[[;;,2]];
rgcm=RandomSample[gcm];
robser=RandomSample[obser];
seq=RandomSample[Range[Length[training]]][[1;;Length[validation]]];

Table[Block[{matrix,rtrainingGCM,rtrainingObser,transform,result,correction,ntransform,s1,s2,correctionValidation,correctionTest},
matrix=RandomVariate[CircularRealMatrixDistribution[n]];
rtrainingGCM=gcm.matrix;
rtrainingObser=obser.matrix;
qMapping[source_,target_,interval_:0.01]:=Block[{qsource,qtarget,position},
	qsource=Table[Quantile[source,q],{q,0,1,interval}];
	qtarget=Table[Quantile[target,q],{q,0,1,interval}];
	position=Table[Position[qsource,_?(#>=source[[k]]&)][[1,1]],{k,Length[source]}];
	Table[{{qsource[[Max[position[[i]]-1,1]]],qsource[[position[[i]]]]},
               {qtarget[[Max[position[[i]]-1,1]]],qtarget[[position[[i]]]]},source[[i]]},{i,Length[position]}]];
transform=ParallelTable[Block[{},Print[i];qMapping[rtrainingGCM[[;;,i]],rtrainingObser[[;;,i]]]],{i,724}];

result=ParallelTable[If[transform[[dim,date]][[1,2]]!=transform[[dim,date]][[1,1]],
(transform[[dim,date]][[3]]-transform[[dim,date]][[1,1]])/(transform[[dim,date]][[1,2]]-transform[[dim,date]][[1,1]])*(transform[[dim,date]][[2,2]]-transform[[dim,date]][[2,1]])+transform[[dim,date]][[2,1]],transform[[dim,date]][[2,1]]],
{date,Length[rtrainingGCM]},{dim,n}];
correction=result.Inverse[matrix];
(*correction=correction/. x_ /; x<0.->0.;*)
(*
energy=2*Total[Flatten[Table[EuclideanDistance[correction[[i]],obser[[j]]],{i,Length[correction]},{j,Length[obser]}]]]/Length[correction]/Length[obser]-Total[Flatten[Table[EuclideanDistance[obser[[i]],obser[[j]]],{i,Length[correction]},{j,Length[obser]}]]]/Length[correction]/Length[obser]-Total[Flatten[Table[EuclideanDistance[correction[[i]],correction[[j]]],{i,Length[correction]},{j,Length[obser]}]]]/Length[correction]/Length[obser];
Print[energy];
*)
energyTraining=2*Total[Flatten[Table[EuclideanDistance[correction[[i]],obser[[j]]],{i,seq},{j,seq}]]]/Length[seq]/Length[seq]-Total[Flatten[Table[EuclideanDistance[obser[[i]],obser[[j]]],{i,seq},{j,seq}]]]/Length[seq]/Length[seq]-Total[Flatten[Table[EuclideanDistance[correction[[i]],correction[[j]]],{i,seq},{j,seq}]]]/Length[seq]/Length[seq];

Print[energyTraining];


ntransform=Table[Sort[DeleteDuplicates[transform[[dim,;;,1;;2]]],#1[[1,2]]<=#2[[1,2]]&],{dim,Length[transform]}];
s1=test[[;;,1]].matrix;
s2=ParallelTable[Block[{raw=s1[[;;,dim]],po},
 result=Map[Block[{ele=#},If[ele<=ntransform[[dim,1,1,1]],ntransform[[dim,2,1,1]],
       If[ele>=ntransform[[dim,-1,1,2]],ntransform[[dim,-1,2,2]],
        Block[{po=ntransform[[dim,Position[ntransform[[dim,;;,1,1]],_?(#<=ele&)][[-1,1]]]]},
          (ele-po[[1,1]])*(po[[1,2]]-po[[1,1]])+po[[2,1]]]]]]&,raw];
 result],{dim,Dimensions[s1][[2]]}];
correctionTest=Transpose[s2].Inverse[matrix];
(*correctionTest=correctionTest/. x_ /; x<0.->0;*)

energyTest=2*Total[Flatten[Table[EuclideanDistance[correctionTest[[i]],test[[j,2]]],{i,Length[test]},{j,Length[test]}]]]/Length[test]/Length[test]-Total[Flatten[Table[EuclideanDistance[test[[i,2]],test[[j,2]]],{i,Length[test]},{j,Length[test]}]]]/Length[test]/Length[test]-Total[Flatten[Table[EuclideanDistance[correctionTest[[i]],correctionTest[[j]]],{i,Length[test]},{j,Length[test]}]]]/Length[test]/Length[test];
Print[energyTest];

s1=validation[[;;,1]].matrix;
s2=ParallelTable[Block[{raw=s1[[;;,dim]],po},
 result=Map[Block[{ele=#},If[ele<=ntransform[[dim,1,1,1]],ntransform[[dim,2,1,1]],
       If[ele>=ntransform[[dim,-1,1,2]],ntransform[[dim,-1,2,2]],
        Block[{po=ntransform[[dim,Position[ntransform[[dim,;;,1,1]],_?(#<=ele&)][[-1,1]]]]},
          (ele-po[[1,1]])*(po[[1,2]]-po[[1,1]])+po[[2,1]]]]]]&,raw];
 result],{dim,Dimensions[s1][[2]]}];
correctionValidation=Transpose[s2].Inverse[matrix];
(*correctionValidation=correctionValidation/. x_ /; x<0.->0;*)
energyValidation=2*Total[Flatten[Table[EuclideanDistance[correctionValidation[[i]],validation[[j,2]]],{i,Length[validation]},{j,Length[validation]}]]]/Length[validation]/Length[validation]-Total[Flatten[Table[EuclideanDistance[validation[[i,2]],validation[[j,2]]],{i,Length[validation]},{j,Length[validation]}]]]/Length[validation]/Length[validation]-Total[Flatten[Table[EuclideanDistance[correctionValidation[[i]],correctionValidation[[j]]],{i,Length[validation]},{j,Length[validation]}]]]/Length[validation]/Length[validation];
Print[energyValidation];

Print[{"Step ",kk,energyTraining,energyValidation,energyTest}];
Export["/usr/workspace/pan11/CycleGAN/MBCn/Step_"<>ToString[kk]<>"_noN.mx",
<|"transform"->transform,
  "matrix"->matrix,
  "energy"->{energyTraining,energyValidation,energyTest}|>];
Set[gcm,correction];],{kk,30}];
