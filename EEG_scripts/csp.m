function [W Cvar]=csp(xdata,varargin)
% CSP Computes the CSP Projection Matrix W from EEG data
%   by Ang Kai Keng (kkang@i2r.a-star.edu.sg)
%
%   Syntax:
%     W=csp(xdata)
%   where
%     xdata: extracted eegdata
%
%   See also extracteegdata.

nch=size(xdata.x,2);
ntrials=size(xdata.x,3);
nclass=2;
leavesingle=false;
eigmethod = 'eig';
covmethod = 'Ramoser';
compute_var=false;

if nargout==2
    compute_var=true;
    Cvar=[];
end
while ~isempty(varargin)
    if ischar(varargin{1})
        switch varargin{1}
            case 'leavesingle'
                leavesingle=true;
                varargin(1)=[];
            case 'svd'
                eigmethod='svd';
                varargin(1)=[];
            case 'eig'
                eigmethod='eig';
                varargin(1)=[];
            case 'blankertz'
                covmethod='Blankertz';
                varargin(1)=[];
            otherwise
                varargin(1)=[];
        end
    end
end

% Prealloc variables
C=zeros(nch,nch,ntrials);
Cmean=cell(nclass);
for i=1:nclass
    Cmean{i}=zeros(nch,nch);
end

switch covmethod
    case 'Ramoser'
        % Ramoser equation (1)
        for trial=1:ntrials
            E=xdata.x(:,:,trial)'; % Transpose samples x channels to channels x samples
            tmpC = (E*E');
            epsilon = 1e-8;
            C(:,:,trial) = tmpC ./ (trace(tmpC) + epsilon);
        end
        for i=1:nclass
            % Important check: Ensure there are trials for this class
            if sum(xdata.y==i) > 0
                Cmean{i}=mean(C(:,:,(xdata.y==i)),3);
            else
                % If no trials, create a zero matrix to avoid NaN from mean of empty set
                Cmean{i} = zeros(nch, nch); 
            end
            
            if compute_var
                Cvar{i}=var(C(:,:,(xdata.y==i)),0,3);
            end
        end
    case 'Blankertz'
        for i=1:nclass
            E=num2cell(xdata.x(:,:,xdata.y==(i)),[1 2]);
            X=cat(1,E{:});
            tmpC=X'*X;
            Cmean{i}=tmpC./trace(tmpC);
        end
end


% --- START: DEBUGGING BLOCK ---
% This block will check the variables right before the error occurs.
fprintf('\n--- Entering CSP Debug Block ---\n');
fprintf('Checking variables before the Ccompo calculation...\n');

num_trials_class1 = sum(xdata.y==1);
num_trials_class2 = sum(xdata.y==2);

fprintf('Number of trials for Class 1 (y==1): %d\n', num_trials_class1);
fprintf('Number of trials for Class 2 (y==2): %d\n', num_trials_class2);

has_nan_in_Cmean1 = any(isnan(Cmean{1}(:)));
has_nan_in_Cmean2 = any(isnan(Cmean{2}(:)));

fprintf('Does Cmean{1} contain NaN? -> %s\n', string(has_nan_in_Cmean1));
fprintf('Does Cmean{2} contain NaN? -> %s\n', string(has_nan_in_Cmean2));

if has_nan_in_Cmean1 || has_nan_in_Cmean2 || num_trials_class1 == 0 || num_trials_class2 == 0
    fprintf('\n*** PROBLEM DETECTED! ***\n');
    fprintf('Pausing execution. Type ''dbquit'' to exit debug mode.\n');
    keyboard; % This command will pause the code and give you the K>> prompt
end
fprintf('--- Exiting CSP Debug Block ---\n\n');
% --- END: DEBUGGING BLOCK ---


% Ramoser equation (2)
Ccompo=Cmean{1}+Cmean{2};

% Find rank of Ccompo (new step to eliminate matrix singularity at line 53)
if leavesingle
    Crank=max(size(Ccompo));
else
    Crank=rank(Ccompo);
end


switch eigmethod
    case 'eig'
        [Ualter,Lalter]=eig(Ccompo,Cmean{1});
        % Sort eigenvalues and eigenvectors
        [Lalter,ind]=sort(diag(Lalter),'descend');
        Ualter=Ualter(:,ind);
        % Retain up to rank eigenvalues and eigenvectors (new step to eliminate matrix singularity at line 53)
        Ualter=Ualter(:,1:Crank);
        W=Ualter';
    case 'svd'
        [U,S,V] = svd(Ccompo);    %#ok<NASGU> %the eigenvalues are in decreasing order
        eigenvalues = diag(S);
        eigenvalues = eigenvalues(1:Crank);
        U = U(:,1:Crank);
        eigenvalues = sqrt(eigenvalues);
        eigenvalues = 1./eigenvalues;
        P=diag(eigenvalues)*U';
        S1 = P*Cmean{1}*P';
        %S2 = P*Cmean{2}*P';
        [u1,eigen1,v1]=svd(S1); %#ok<NASGU>
        %[ub,eigenB,vb]=svd(S2);
        W = u1'*P; % get the projection matrix
end

%Normalize projection matrix W
for i=1:length(W) %W is a square matrix
    W(i,:)=W(i,:)./norm(W(i,:));
end
% Common spatial patterns in inverse of W
%A=pinv(W); % Common spatial patterns
end