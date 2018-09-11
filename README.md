# baseballhof-macos
Visual Studio for Mac Console project using ML.NET .NET Core library to predict Baseball Hall Of Fame Induction for batters.

Project Uses:
- Selected historical baseball batting data from 1900-2017
- Visual Studio for Mac (macOS) 7.6.x
- ML.NET Package (Machine Learning .NET library), current version v0.5
- .NET Core 2.1 SDK

Supervised Learning Setup: A binary classifier is used to train a yes/no prediction and persist a model.  The model is used to predict HOF induction on a validation set, which outputs a list of metrics and correct/incorrect predictions.

![Visual Studio macOS](https://github.com/bartczernicki/baseballhof-macos/blob/master/MachineLearningBaseBallHOF/ProjectInVisualStudio.png)
