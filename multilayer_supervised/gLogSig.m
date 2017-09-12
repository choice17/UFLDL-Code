function grad = gLogSig(a)
%compute grad of a = logsig(z);
grad = a.*(1-a);
end