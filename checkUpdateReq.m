function [ theta,thetaVec,totaldiff1,flag ] = checkUpdateReq( trkptsdiff1,medianValue )
%This function checks if ReInitialization of Tracking landmarks is required
%   Detailed explanation goes here
% for i=1:numel(x)
%     for j=1:i
%         
% thetaVec(i,j)=x(i)+y(j);
%     end
% end


%     for j=6:f
        
% thetaVec=subspace(basis1,tmplbasis{f});
% totaldiff1=myBasisdiffFunc(basis1,tmplbasis{f});
totaldiff1=0;
% totaldiff1(1,j)=myBasisdiffFunc(tmplbasis{f},tmplbasis{j});
%     end
thetaVec=0;
theta=thetaVec;
%       flag=0;
% % if(totaldiff1>150)
       if( trkptsdiff1>medianValue)


           flag=1;
       else
           flag=0;
       end
% % 
end


 