@CLAUDE.md Let's get rid of MonitorModeId::kEstLoad.

Instead, let's add a new method to each task. Call this method GetPerfFeatures. 

A new method should be added to the Container class called GetPerfFeatures. Add this method to the chi_refresh_repo autogeneration functions. In each class, this will cast a generic task to concrete task 
type and then call GetPerfFeatures. The input to GetPerfFeatures is a struct called Sample. Sample has a 
method named AddFeature, which has overrides for string and float. 

Bdev, for example, can choose a linear model