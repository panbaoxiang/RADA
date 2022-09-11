size={32,32};
c=32;
depth=3;

res[c_]:=NetGraph[<|"long"->Flatten[Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],NormalizationLayer[],Ramp},2]][[1;;-2]],
          "plus"->TotalLayer[],
          "short"->ConvolutionLayer[c,{1,1}]|>,
   {NetPort["Input"]->"long"->"plus",NetPort["Input"]->"short"->"plus"}]

upres[c_,size_]:=NetGraph[<|"long"->{NormalizationLayer[],Ramp,ResizeLayer[size],ConvolutionLayer[c,{3,3},"PaddingSize"->1],
	                                NormalizationLayer[],Ramp,ConvolutionLayer[c,{3,3},"PaddingSize"->1]},
          "plus"->TotalLayer[],
          "short"->{ResizeLayer[size],ConvolutionLayer[c,{1,1}]}|>,
   {NetPort["Input"]->"long"->"plus",NetPort["Input"]->"short"->"plus"}]

contract[channel_,crop_:{{1,1},{1,1}}]:=NetGraph[{"conv"->res[channel],"pooling"->PoolingLayer[2,2,"Function"->Mean],
                                   "cropping"->PartLayer[{;;,crop[[1,1]];;-crop[[1,-1]],crop[[2,1]];;-crop[[2,-1]]}]},
                         {NetPort["Input"]->"conv"->"pooling"->NetPort["Pooling"],"conv"->"cropping"->NetPort["Shortcut"]}];

expand[channel_,size_]:=NetGraph[{"deconv"->upres[channel,size],
                            "join"->CatenateLayer[],
                            "conv"->res[channel/2]},
                         {NetPort["Input"]->"deconv"->"join",
                          NetPort["Shortcut"]->"join"->"conv"}];

UNet=NetGraph[<|Table[{"contract_"<>ToString[i]->contract[c*2^(i-1)],
					  "expand_"<>ToString[i]->expand[c*2^(i-1),size/2^(i-1)]},{i,depth}],
					  "ubase"->res[c*(depth+1)]|>,  
Flatten[Table[{NetPort["contract_"<>ToString[i],"Pooling"]->If[i<depth,"contract_"<>ToString[i+1],"ubase"->NetPort["expand_"<>ToString[depth],"Input"]],
 NetPort["contract_"<>ToString[i],"Shortcut"]->NetPort["expand_"<>ToString[i],"Shortcut"],
 NetPort["expand_"<>ToString[i],"Output"]->If[i>1,NetPort["expand_"<>ToString[i-1],"Input"],NetPort["Output"]]},{i,depth}]]]
