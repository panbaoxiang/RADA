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

seq=Flatten[{Select[Flatten[Table[Range[(i-1)*365+1,i*365],{i,Select[Range[1,60],Mod[#,10]<=7&]}]],#<=length+vlength&],
            Select[Flatten[Table[Range[(i-1)*365+1,i*365],{i,Select[Range[1,60],Mod[#,10]>7&]}]],#<=length+vlength&]}];
{nP4GCM,ndynamics4GCM,nP4Obser,ndynamics4Obser}=Map[#[[seq]]&,{nP4GCM,ndynamics4GCM,nP4Obser,ndynamics4Obser}];

training=Table[<|"P_GCM"->nP4GCM[[i]],
             "D_GCM"->ndynamics4GCM[[i]],
             "P_Obser"->nP4Obser[[i]],
             "D_Obser"->ndynamics4Obser[[i]]
             |>,{i,length}];

DADT=Table[If[validMatrix[[i,j]]==0,
	Table[0,Length[test]],
	Block[{sgcm=training[[;;,"P_GCM"]][[;;,1,i,j]],
	       sobser=training[[;;,"P_Obser"]][[;;,1,i,j]],position,threshold,a1,range},
	  position=Position[Sort[sobser],0.][[-1,1]];
	  threshold=Sort[sgcm][[position]];
	  a1=sgcm/.x_/;x<=threshold->0;
	  range=Block[{x=Sort[a1],y=Sort[sobser]},{Table[Quantile[x[[position+1;;]],i],{i,0,1,0.01}],Table[Quantile[y[[position+1;;]],i],{i,0,1,0.01}]}];
	Table[Block[{element,p},
		element=test[[kk,"P_GCM",1,i,j]];
		p=Position[range[[1]],_?(#<element&)];
	  If[Length[p]==0,0,(element-range[[1,p[[-1,1]]]])/(range[[1,Min[p[[-1,1]]+1,Length[range[[1]]]]]]-range[[1,p[[-1,1]]]]+10^-4)*
		(range[[2,Min[p[[-1,1]]+1,Length[range[[1]]]]]]-range[[2,p[[-1,1]]]]+10^-4)+range[[2,p[[-1,1]]]]]],{kk,Length[test]}]]],
{i,Length[validMatrix]},{j,Dimensions[validMatrix][[2]]}];
DADT=Transpose[DADT,{2,3,1}];
GCM=test[[;;,"P_GCM"]][[;;,1]];
OBSER=test[[;;,"P_Obser"]][[;;,1]];
{DADT,GCM,OBSER}=Map[Exp[#]-1&,{DADT,GCM,OBSER}];
Export["/g/g92/pan11/result_Baseline_QuantileMapping.mx",Map[NumericArray[#,"Real32"]&,{DADT,GCM,OBSER}]]

