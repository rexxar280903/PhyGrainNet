# Experiment Protocol

Every controlled comparison must hold constant the grouped folds, preprocessing, seed policy, input resolution unless it is the variable being tested, optimizer/scheduler policy where applicable, official metric implementation, and early-stopping rule.

Every run records its experiment ID, date, Git commit, config, fold, seed, parameter count, runtime, primary EMD, secondary metrics, and decision notes.

A result may influence final model selection or manuscript claims only when it passes the active readiness gate and is reproducible from committed code and configuration.
