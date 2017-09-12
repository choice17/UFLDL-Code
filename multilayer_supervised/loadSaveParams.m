function [opt_params,params] = loadSaveParams(Filename)
load(Filename);
opt_params = theta;
params = theta;
