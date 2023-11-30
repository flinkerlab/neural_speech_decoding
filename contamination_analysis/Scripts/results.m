
addpath(genpath('/Users/james/contamination/Toolbox/'));
set(gcf,'visible','off')
%% Global parameters
DIR = {'NY717', 'NY718', 'NY869', 'NY857', 'NY786', 'NY720', 'NY744', 'NY758','NY837', 'NY733', 'NY741', 'NY751', 'NY802', 'NY756', 'NY723','NY781', 'NY830', 'NY721', 'NY753', 'NY863', 'NY763', 'NY743','NY838', 'NY847', 'NY794', 'NY757', 'NY828', 'NY797', 'NY787','NY799', 'NY708', 'NY844', 'NY789', 'NY748', 'NY722', 'NY746','NY810', 'NY841', 'NY782', 'NY704', 'NY836', 'NY737', 'NY791','NY739','NY742', 'NY749', 'NY798', 'NY829'};
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
