% Feature selection from W, CSP matrix
% ChRanking shows the channels rank 
% numchannels is number of the selected channels

function [producedfeatur,selectedw]=featcrossval(finaldataset,ChRanking,numchannel)
for i=1:numchannel
    a.x(:,i,:)=finaldataset.x(:,ChRanking(1,i),:);
end

a.y=finaldataset.y;
%W=cspbci4(a);
W=csp(a);
if numchannel>6
    selectedw=[W(1,:);W(2,:);W(3,:);W(end-2,:);W(end-1,:);W(end,:)];
%selectedw=[W(1,:);W(2,:);W(end-1,:);W(end,:)];
else
    selectedw=[W(1,:);W(end,:)];
end
ntrial=size(finaldataset.x,3);
for trial=1:ntrial
    selectedZ=selectedw*(a.x(:,:,trial))';
    % producedfeature is features*number of channel*number of trials
    producedfeatur.x(:,trial)=log(var( selectedZ(:,:)')./sum(var( selectedZ(:,:)')));
   
end
producedfeatur.y=finaldataset.y;

