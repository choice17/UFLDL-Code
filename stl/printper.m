function msgL = printper(i,N,per,msgL)

M=N/per;
if i<M+1
if i/floor(M)==1
  len = toc;
  fprintf('\t estimate total time needed: %d seconds\t\n',floor(len*100));
  fprintf('\t\t\t');
elseif i==1
    tic;
end
end

if mod(i,floor(M))==0 
    back = strcat(repmat('\b',1,msgL));
    fprintf(back);
    msg = strcat(string(floor(i/(M)*100/per)),'%%\tprocessing\n');
    msgL = length(char(msg))-3;
    fprintf(msg);
    
end