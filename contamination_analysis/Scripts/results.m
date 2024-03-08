
addpath(genpath('/Users/james/contamination/Toolbox/'));
set(gcf,'visible','off')
%% Global parameters
DIR = { 'HB02'};
pvalue = []
for j = 1:length(DIR)
    sample = DIR{j}
    disp(sample)
    try
        load(['/Users/james/contamination/', 'Analysis results/',DIR{j},'/',DIR{j},'_object.mat'])
    
        disp(obj.criterion_value)
        pvalue = [pvalue, obj.criterion_value]
    
    
    
    
    
    
    catch exception
        disp([DIR{j} ,'failed\n'])
        continue; 
    end
end

disp(pvalue)
save('Analysis results/pvalue.mat','pvalue')
