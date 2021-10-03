urange = 10Pi; vrange = 1.5; t = 10;rec=6; dis=Sqrt[5]-1;
f[u_, v_] = #/Sqrt[# . #] &@{Cos[u], Sin[u], (u + v)/t};
{r, \[Theta], \[CurlyPhi]} = ToSphericalCoordinates[{x, y, z}] /. Thread[{x, y, z} -> f[u, v]] //FullSimplify;
map1=Rasterize[Block[{range=ArcSin[f[urange,vrange][[-1]]]/Pi*180.},
	GeoGraphics[,GeoBackground->"Satellite",GeoRange->{{-range, range}, {-180., 180.}},
	PlotRangePadding->None,ImagePadding->None,Frame->False,Axes->False,GeoRangePadding->None,GeoZoomLevel->4]],RasterSize->5000];
	
{sgrid,ogrid}=Block[{result,lat,lon,grid,testpoint,poly},
	lat=Range[-90,90,1];
	lon=Range[-180,180,1];
	grid=Flatten[Table[{lat[[i]],lon[[j]]},{i,Length[lat]},{j,Length[lon]}],1];
	testpoint[poly_, pt_] := And[pt[[1]]>Min[poly[[;;,1]]],pt[[1]]<Max[poly[[;;,1]]],
                             pt[[2]]>Min[poly[[;;,2]]],pt[[2]]<Max[poly[[;;,2]]],
	Round[(Total@ Mod[(# - RotateRight[#]) &@(ArcTan @@ (pt - #) & /@ poly), 2 Pi, -Pi]/2/Pi)] != 0];
	poly=Flatten[EntityClass["GeographicRegion","Continents"]["Polygon"][[;;,1,1]],1];
	result=Table[If[Mod[i,1000]==0,Print[i]];Map[testpoint[#,grid[[i]]]&,poly]/.List->Or,{i,Length[grid]}];
	{grid[[Position[result,True][[;;,1]]]],grid[[Position[result,False][[;;,1]]]]}];
(*Export["/Users/pan11/Documents/PROUD/Logo/sgrid.mx",<|"continent"\[Rule]sgrid,"ocean"\[Rule]ogrid|>]*)
(*{sgrid,ogrid}=Values[Import["/Users/pan11/Documents/PROUD/Logo/sgrid.mx"]];*)

map2=Rasterize[Block[{range=ArcSin[f[urange,vrange][[-1]]]/Pi*180.},
GeoGraphics[{Table[GeoMarker[GeoPosition[sgrid[[i]]],Style[RandomSample[{"0","1"}][[1]],
{2,Bold,Opacity[RandomReal[{.2,1}]],RGBColor[{0.18,0.443,0.13}],FontFamily->"Silom"}]],{i,1,Length[sgrid]}],
			Table[GeoMarker[GeoPosition[ogrid[[i]]],Style[RandomSample[{"0","1"}][[1]],
{2,Bold,Opacity[RandomReal[{.2,1}]],RGBColor[{0.03,0.09,0.21}],FontFamily->"Silom"}]],{i,1,Length[ogrid]}]},
	GeoBackground->{RGBColor[{0.03,0.09,0.21}],Opacity[0.9]},
	GeoRange->{{-range, range}, {-180., 180.}},
	PlotRangePadding->None,ImagePadding->None,Frame->False,Axes->False,GeoRangePadding->None,GeoZoomLevel->4]],RasterSize->5000];

left=ParametricPlot3D[f[u, v], {u, -urange+Pi/2, urange-Pi/2}, {v, -vrange, vrange}, 
 Mesh->None,
 Lighting->"Neutral",
 MaxRecursion -> rec, 
 Axes -> False, Boxed -> False, TextureCoordinateScaling -> False, 
 TextureCoordinateFunction -> Function[{x, y, z, u, v}, {\[CurlyPhi]/(2 \[Pi]), -\[Theta]/\[Pi]}],
 PlotStyle -> Texture[map1],
 PlotPoints -> 30,PlotRange->Full,ImageSize->400,Lighting->"Neutral"];
right=Block[{tempt},
 tempt=ParametricPlot3D[f[u, v]*{1,1,-1}, {u, -urange-Pi/2,urange+Pi/2}, {v, -vrange, vrange}, 
 Mesh->None,
 Lighting->"Neutral",
 MaxRecursion -> rec, 
 Axes -> False, Boxed -> False, TextureCoordinateScaling -> False, 
 TextureCoordinateFunction -> Function[{x, y, z, u, v}, {\[CurlyPhi]/(2 \[Pi]), \[Theta]/\[Pi]}],
 PlotStyle -> Texture[map2],
 PlotPoints -> 30,PlotRange->Full];
 Graphics3D[Translate[First@tempt,{dis, 0, 0}],ImageSize->400]];
joint=ParametricPlot3D[
{x*(f[urange-Pi/2, v+2vrange]+{dis,0,0}-f[urange-Pi/2, v])+f[urange-Pi/2, v],
 x*(f[-urange+Pi/2, v-2vrange]+{dis,0,0}-f[-urange+Pi/2, v])+f[-urange+Pi/2, v]},
{x,0,1},{v,-vrange,vrange},MaxRecursion->rec,Mesh->None,PlotStyle->{RGBColor[(*{0.125,0.164,0.07}*){0.03,0.09,0.21}],RGBColor[{0.03,0.09,0.21}]}];
result=Show[{left,right,joint},PlotRange->Full,ViewPoint->{1, -2, 1},ImageSize->700]
