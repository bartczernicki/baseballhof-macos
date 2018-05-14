# baseballhof-macos
Visual Studio for Mac Console project using ML.NET .NET Core library to predict Baseball Hall Of Fame Induction for batters.

Project Uses:
- Selected historical baseball batting data from 1900-2015
- Visual Studio for Mac (macOS) 7.5
- ML.NET Package (Machine Learning .NET library)
- .NET Core 2.x SDK

Supervised Learning Setup: A binary classifier is used to train a yes/no prediction and persist a model.  The model is used to predict HOF induction on a validation set, which outputs a list of metrics and correct/incorrect predictions.

![Visual Studio macOS](https://github.com/bartczernicki/baseballhof-macos/blob/master/MachineLearningBaseBallHOF/ProjectInVisualStudio.png)
