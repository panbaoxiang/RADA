vateStatisticsCorrection=Table[Block[{p=RandomReal[{0,1}]},
	ChoiceDialog[If[p<1/2,plot[dadt[[RandomSample[Range[Length[dadt]]][[1]]]],plat4GCM,plon4GCM], 
			     plot[obser[[RandomSample[Range[Length[obser]]][[1]]]],plat4GCM,plon4GCM]],
		{"Observation"->If[p>=1/2,{1,1},{1,0}],
		 "Simulation"->If[p<1/2,{0,0},{0,1}]}]],{i,50}];

voteStatisticsRaw=Table[Block[{p=RandomReal[{0,1}]},
	ChoiceDialog[If[p<1/2,plot[gcm[[RandomSample[Range[Length[gcm]]][[1]]]],plat4GCM,plon4GCM], 
			     plot[obser[[RandomSample[Range[Length[obser]]][[1]]]],plat4GCM,plon4GCM]],
		{"Observation"->If[p>=1/2,{1,1},{1,0}],
		 "Simulation"->If[p<1/2,{0,0},{0,1}]}]],{i,50}];
Export["/Users/pan11/Documents/CycleGAN/Results/Pamler_Turing_Raw_"<>CreateUUID[]<>".mx",voteStatisticsRaw];
