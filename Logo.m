urange = 10Pi; vrange = 1.5; t = 10;rec=5; dis=Sqrt[5]-1;
f[u_, v_] = #/Sqrt[# . #] &@{Cos[u], Sin[u], (u + v)/t};
{r, \[Theta], \[CurlyPhi]} = ToSphericalCoordinates[{x, y, z}] /. Thread[{x, y, z} -> f[u, v]] //FullSimplify;
map1=Rasterize[Block[{range=ArcSin[f[urange,vrange][[-1]]]/Pi*180.},
	GeoGraphics[,GeoBackground->"Satellite",GeoRange->{{-range, range}, {-180., 180.}},
	PlotRangePadding->None,ImagePadding->None,Frame->False,Axes->False,GeoRangePadding->None,GeoZoomLevel->4]],RasterSize->5000];
map2=ImageEffect[map1,"EdgeStylization"];
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
{x,0,1},{v,-vrange,vrange},MaxRecursion->rec,Mesh->None,PlotStyle->{RGBColor[{0.125,0.164,0.07}],RGBColor[{0.03,0.09,0.21}]}];
result=Show[{left,right,joint},PlotRange->Full,ViewPoint->{1, -2, 1},ImageSize->700]
