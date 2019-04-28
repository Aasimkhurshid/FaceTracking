function isgood = update_check_Dict(pred,thr, ind)
% checks if prediction error doesnt exeeds limit
%  I can use median, to check

% if pred.cls_err(ind) > 1 || pred.rec_err(ind) < 0.35
if  pred <=  median(thr)
    isgood = 1;
else
    isgood=0;
end

end

