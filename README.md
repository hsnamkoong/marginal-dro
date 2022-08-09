# Code release for "Distributionally Robust Losses for Latent Covariate Mixtures"

This repository contains the loss function and support code for the paper "Distributionally Robust Losses for Latent Covariate Mixtures"

The release consists of two files that each contain the distributionally robust dual and wrapper code to support bisection search.

- The file `dual_lip_risk_bound` contains pytorch modules for the dual, covariate shift DRO losses. These can be used as loss function wrappers after fixing the Lipschitz smoothness L/epsilon
- The file `utils` contains other utilities such as 
- `environment.yml` contains a copy of the working env for this project. It may also include unecessary packages.

For any questions or issues, please contact Tatsunori Hashimoto (thashim@stanford.edu)
          
          
# License 
Copyright (C) 2022 Tatsunori Hashimoto

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
