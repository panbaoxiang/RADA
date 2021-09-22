hype={RandomSample[{300,500,1000}][[1]],RandomSample[{3,5,7,10}][[1]],RandomSample[Range[1,3]][[1]]};
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

validation=Table[<|"P_GCM"->nP4GCM[[i]],
             "D_GCM"->ndynamics4GCM[[i]],
             "P_Obser"->nP4Obser[[i]],
             "D_Obser"->ndynamics4Obser[[i]]
	     |>,{i,length,length+vlength}];


generator=NetGraph[<|"Catenate"->CatenateLayer[1],
                    "Padding"->PaddingLayer[{{0,0},{5,5},{4,4}}],
                    "chain"->{ConvolutionLayer[64,{3,3}],
          BatchNormalizationLayer[],
          Ramp,
	  ConvolutionLayer[128,{3,3}],
          BatchNormalizationLayer[],
          Ramp, 
	  ConvolutionLayer[256,{3,3}],
          BatchNormalizationLayer[],
	  Ramp,
          ConvolutionLayer[512,{5,3}],
          BatchNormalizationLayer[],
          Ramp,
          ConvolutionLayer[1,{1,1}]},
          "combine"->ThreadingLayer[Plus],
          "cut"->{ConstantTimesLayer["Scaling"->{validMatrix},LearningRateMultipliers->0.],Ramp}
          |>,
     {NetPort["P"]->"Padding"->"Catenate",
      NetPort["z"]->"Catenate"->"chain"->"combine",
      NetPort["P"]->"combine"->"cut"},
    "P"->dim,
    "z"->dim2]

generatorGCM2Obser = NetInsertSharedArrays[generator, "generatorGCM2Obser/"];
cycleGCM2Obser = NetInsertSharedArrays[generator, "generatorGCM2Obser/"];
generatorGCM2ObserR=  NetInsertSharedArrays[generator, "generatorGCM2Obser/"];

generatorObser2GCM = NetInsertSharedArrays[generator, "generatorObser2GCM/"];
cycleObser2GCM = NetInsertSharedArrays[generator, "generatorObser2GCM/"];
generatorObser2GCMR = NetInsertSharedArrays[generator, "generatorObser2GCM/"];

discriminator = NetChain[{
    ConvolutionLayer[16,{3,3},"Stride"->1],BatchNormalizationLayer[], Ramp,
    ConvolutionLayer[32,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[128,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,
    FlattenLayer[],BatchNormalizationLayer[],hype[[1]],Ramp,BatchNormalizationLayer[], 1, ElementwiseLayer["HardSigmoid"]},
  "Input" -> dim];
discriminatorGCM2Obser = discriminator;
discriminatorObser2GCM = discriminator;


RdownscaleGCM=Import["/g/g92/pan11/Backup_CycleGAN/Downscaling_GCM.mx"]["net"];
DeltaGCM=Import["/g/g92/pan11/Backup_CycleGAN/Downscaling_GCM.mx"]["mse"];

RdownscaleObser=Import["/g/g92/pan11/Backup_CycleGAN/Downscaling_Obser.mx"]["net"];
DeltaObser=Import["/g/g92/pan11/Backup_CycleGAN/Downscaling_Obser.mx"]["mse"];

cycleGAN =NetGraph[<|
    "Generator_GCM->Obser" -> generatorGCM2Obser,
    "Generator_GCM->Obser_SelfRegression" -> generatorGCM2ObserR,
    "Cycle_GCM->Obser" -> cycleGCM2Obser,
    "Discriminator_GCM->Obser" -> NetMapOperator[discriminatorGCM2Obser],
    "Cat_GCM->Obser" -> CatenateLayer[],
    "Reshape_GCM->Obser" -> ReshapeLayer[Prepend[dim,2]],
    "Flat_GCM->Obser" -> ReshapeLayer[{2}],
    "Fake_GCM->Obser"->PartLayer[1],
    "Real_GCM->Obser"->PartLayer[2],
    "Scale_GCM->Obser" -> ConstantTimesLayer["Scaling" -> {-1, 1},LearningRateMultipliers->0],
    "MS_GCM->Obser"->MeanAbsoluteLossLayer[],

    "Generator_Obser->GCM" -> generatorObser2GCM,
    "Generator_Obser->GCM_SelfRegression" -> generatorObser2GCMR,
    "Cycle_Obser->GCM" -> cycleObser2GCM,
    "Discriminator_Obser->GCM" -> NetMapOperator[discriminatorObser2GCM],
    "Cat_Obser->GCM" -> CatenateLayer[],
    "Reshape_Obser->GCM" -> ReshapeLayer[Prepend[dim,2]],
    "Flat_Obser->GCM" -> ReshapeLayer[{2}],
    "Fake_Obser->GCM"->PartLayer[1],
    "Real_Obser->GCM"->PartLayer[2],
    "Scale_Obser->GCM" -> ConstantTimesLayer["Scaling" -> {-1, 1},LearningRateMultipliers->0],
    "MS_Obser->GCM"->MeanAbsoluteLossLayer[],

    "R_Downscaling_GCM"->RdownscaleGCM,
    "R_Downscaling_Obser"->RdownscaleObser,

    "MS_GCM_RDownscaling"->MeanSquaredLossLayer[],
    "MS_Obser_RDownscaling"->MeanSquaredLossLayer[],
    "Max_GCM_RDownscaling"->ElementwiseLayer[Max[#,DeltaGCM]-DeltaGCM &],
    "Max_Obser_RDownscaling"->ElementwiseLayer[Max[#,DeltaObser]-DeltaObser &],

    "MS_GCM2Obser_SelfRegression"->MeanAbsoluteLossLayer[],
    "MS_Obser2GCM_SelfRegression"->MeanAbsoluteLossLayer[]
    |>,

   {NetPort["P_GCM"] ->NetPort["Generator_GCM->Obser","P"],
    NetPort["D_GCM"]->NetPort["Generator_GCM->Obser","z"],
    "Generator_GCM->Obser"->"Cat_GCM->Obser",
    NetPort["P_Obser"] -> "Cat_GCM->Obser",
    "Cat_GCM->Obser" -> "Reshape_GCM->Obser" -> "Discriminator_GCM->Obser" -> "Flat_GCM->Obser" -> "Scale_GCM->Obser" ->
    "Fake_GCM->Obser"->NetPort["FakeLoss_GCM->Obser"],
    "Scale_GCM->Obser"->"Real_GCM->Obser"->NetPort["RealLoss_GCM->Obser"],
    "Generator_GCM->Obser"->NetPort["Cycle_Obser->GCM","P"],
    NetPort["D_GCM"]->NetPort["Cycle_Obser->GCM","z"],
    "Cycle_Obser->GCM"->"MS_GCM->Obser",
    NetPort["P_GCM"]->"MS_GCM->Obser"->NetPort["ReconstructionLoss_GCM->Obser"],

    NetPort["D_GCM"]->"R_Downscaling_Obser"->"MS_Obser_RDownscaling",
    "Generator_GCM->Obser"->"MS_Obser_RDownscaling"->"Max_Obser_RDownscaling"->NetPort["Loss_RDownscaling_GCM"],
    NetPort["D_Obser"]->"R_Downscaling_GCM"->"MS_GCM_RDownscaling",
    "Generator_Obser->GCM"->"MS_GCM_RDownscaling"->"Max_GCM_RDownscaling"->NetPort["Loss_RDownscaling_Obser"],

    NetPort["P_Obser"] ->NetPort["Generator_Obser->GCM","P"],
    NetPort["D_Obser"] ->NetPort["Generator_Obser->GCM","z"],
    "Generator_Obser->GCM"-> "Cat_Obser->GCM",
    NetPort["P_GCM"] -> "Cat_Obser->GCM",
    "Cat_Obser->GCM" -> "Reshape_Obser->GCM" -> "Discriminator_Obser->GCM" -> "Flat_Obser->GCM" -> "Scale_Obser->GCM" ->
    "Fake_Obser->GCM"->NetPort["FakeLoss_Obser->GCM"],
    "Scale_Obser->GCM"->"Real_Obser->GCM"->NetPort["RealLoss_Obser->GCM"],
    "Generator_Obser->GCM"->NetPort["Cycle_GCM->Obser","P"],
    NetPort["D_Obser"] ->NetPort["Cycle_GCM->Obser","z"],
    "Cycle_GCM->Obser"->"MS_Obser->GCM",
    NetPort["P_Obser"]->"MS_Obser->GCM"->NetPort["ReconstructionLoss_Obser->GCM"],

    NetPort["P_GCM"]->NetPort["Generator_Obser->GCM_SelfRegression","P"],
    NetPort["D_GCM"]->NetPort["Generator_Obser->GCM_SelfRegression","z"],
    "Generator_Obser->GCM_SelfRegression"->"MS_Obser2GCM_SelfRegression",
    NetPort["P_GCM"]->"MS_Obser2GCM_SelfRegression"->NetPort["Loss_Obser2GCM_SelfRegression"],

    NetPort["P_Obser"]->NetPort["Generator_GCM->Obser_SelfRegression","P"],
    NetPort["D_Obser"]->NetPort["Generator_GCM->Obser_SelfRegression","z"],
    "Generator_GCM->Obser_SelfRegression"->"MS_GCM2Obser_SelfRegression",
    NetPort["P_Obser"]->"MS_GCM2Obser_SelfRegression"->NetPort["Loss_GCM2Obser_SelfRegression"]
    },
   "P_Obser" -> dim,
   "P_GCM" -> dim,
   "D_GCM" -> dim2,
   "D_Obser" -> dim2];

DiffMean=0.020;
DiffVar=0.025;


obserMean=Mean[validation[[;;,"P_Obser"]]][[1]];
obserVar=Variance[validation[[;;,"P_Obser"]]][[1]];

index="N"<>ToString[hype[[1]]]<>"_R"<>ToString[hype[[2]]]<>"_B"<>ToString[hype[[3]]]<>"_"<>StringSplit[CreateUUID[],"-"][[1]];
Print[index];
ReportCycleGan2[net_] :=
  Block[{gen,dGCM,dObser,obserG,obserD,gcmD,meanDiff,varDiff,dlossGCM,dlossObser},
        gen=net[["Generator_GCM->Obser"]];
        obserG=Map[gen[<|"P"->#[["P_GCM"]],"z"->#[["D_GCM"]]|>,TargetDevice->"GPU"]&,validation];
        meanDiff=Mean[Abs[Flatten[Mean[obserG][[1]]-obserMean]]];
        varDiff=Mean[Abs[Flatten[Variance[obserG][[1]]-obserVar]]];
        Print[TableForm[{{DiffMean,DiffVar},
               {meanDiff,varDiff}}]];
        If[meanDiff+varDiff<=DiffMean+DiffVar,
          Block[{},
		Print[index];
                Export["/g/g92/pan11/CycleGAN_"<>index<>".mx",net];
                Set[{DiffMean,DiffVar},{meanDiff,varDiff}]]]];

NetTrain[cycleGAN,
    {Function[Block[{base,choice,choice2},
        base=RandomSample[Range[2,length],#BatchSize];
        choice=Map[Block[{daylag=RandomSample[Range[-15,15]][[1]],yearlag=RandomSample[Range[-5,5]][[1]],tempt},
                                tempt=#+daylag+yearlag*365;
                                If[And[tempt>0,tempt<=length],tempt,#]]&,base];
        <|"P_GCM"->nP4GCM[[base]],
          "D_GCM"->ndynamics4GCM[[base]],
          "P_Obser"->nP4Obser[[choice]],
          "D_Obser"->ndynamics4Obser[[choice]]|>]], "RoundLength" -> Length[nP4GCM]},
    LossFunction ->{"FakeLoss_GCM->Obser"->Scaled[1],"RealLoss_GCM->Obser"->Scaled[1],"ReconstructionLoss_GCM->Obser"->Scaled[-hype[[3]]],
                    "FakeLoss_Obser->GCM"->Scaled[1],"RealLoss_Obser->GCM"->Scaled[1],"ReconstructionLoss_Obser->GCM"->Scaled[-hype[[3]]],
                    "Loss_Obser2GCM_SelfRegression"->Scaled[-hype[[3]]],"Loss_GCM2Obser_SelfRegression"->Scaled[-hype[[3]]],
                    "Loss_RDownscaling_GCM"->Scaled[-hype[[3]]],"Loss_RDownscaling_Obser"->Scaled[-hype[[3]]]},
    TrainingUpdateSchedule -> {"Discriminator_GCM->Obser"|"Discriminator_Obser->GCM",
                               "Generator_GCM->Obser"|"Generator_Obser->GCM",
			       "Cycle_GCM->Obser"|"Cycle_Obser->GCM",
  			       "Generator_GCM->Obser_SelfRegression"|"Generator_Obser->GCM_SelfRegression"},
    LearningRateMultipliers -> {"Scale_GCM->Obser" -> 0, "Scale_Obser->GCM" -> 0,
                             "Generator_GCM->Obser" -> -1,"Generator_Obser->GCM"->-1,
                             "Generator_GCM->Obser_SelfRegression" -> -1,"Generator_Obser->GCM_SelfRegression"->-1,
			     "Cycle_GCM->Obser"->-1,"Cycle_Obser->GCM"->-1,
                             "Discriminator_Obser->GCM"->1,"Discriminator_GCM->Obser"->1,
                             "R_Downscaling_GCM"->0,"R_Downscaling_Obser"->0},
    BatchSize -> 32,
    TargetDevice->"GPU",
    MaxTrainingRounds->400,
    Method -> {"ADAM", "Beta1" -> 0.5, "LearningRate" -> 10^-4,
                           "WeightClipping" -> {"Discriminator_Obser->GCM"-> hype[[2]]/100.,"Discriminator_GCM->Obser"->hype[[2]]/100.}},
    TrainingProgressReporting -> {{Function@ReportCycleGan2[#Net], "Interval" -> Quantity[300, "Batches"]},"Print"}]
