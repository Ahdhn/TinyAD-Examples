@echo off
set FILES=../RXMesh/input/ear2.obj ../RXMesh/input/edgar-allan-poe-1.obj ../RXMesh/input/Nefertiti.obj ../RXMesh/input/Lucy_3M_O10.obj ../RXMesh/input/20131208_VenusDeMilo_Full_Scale_captured_by_Cosmo_Wenman.obj ../RXMesh/input/wingedvictory.obj ../RXMesh/input/blue-crab-master-geometry.obj ../RXMesh/input/SI/f1930_54-part_02-smartscan-fixed-textured.obj ../RXMesh/input/SI/bell_x1-complete_with_vane-smooth.obj ../RXMesh/input/SI/npg_70_4_bust-hires_unwrapped.obj ../RXMesh/input/SI/f1930_54-part_01-smartscan-fixed-textured.obj ../RXMesh/input/SI/mammoth-master_model.obj ../RXMesh/input/SI/f1961_33-part_02-x_pol-ort_texture.obj ../RXMesh/input/SI/cosmic_buddha-full_resolution-no_texture.obj

set exe=build\Release\smoothing.exe

for %%F in (%FILES%) do (
    REM echo Processing %%F
    %exe% %%F
)