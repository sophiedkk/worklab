# Worklab: a wheelchair biomechanics mini-package
![DOI](badge.svg)

Essential data analysis and (pre-)processing scripts used in my project researching the [Lode Esseda](https://www.lode.nl/en/product/esseda-wheelchair-ergometer/637) 
wheelchair ergometer in the worklab at the University Medical Centre Groningen. Includes all basic io and calculations for the equipment in the worklab, which means:

* Measurement wheel (Optipush and SMARTwheel) and ergometer (Esseda) data processing
* Push-by-push analysis
* Spirometer (COSMED) data processing
* IMU (NGIMU) data processing
* Kinamatics (Optotrak) data processing
* more(?)

## Rationale
This is an attempt to make analysis of wheelchair biomechanics data more accessible and transparent. Previously all 
analyses were performed with commercial software that is not available to everyone, especially to people not associated 
with a university. Having the analysis in Python makes it accessible and more readable (hopefully) for everyone.
By sharing the code I hope to be transparent and to reduce the amount of times this code has to be written by other people.

## Target audience
People working in our lab that want to work with data from any of our instruments. It can, of course, also be used by other
people, provided that you have similar equipment. Most of the time you will only need one or two functions which you can 
just take from the source code or you can just install the package as it has very little overhead anyways and only uses
packages that you probably already have installed. Also have a look at the 
[examples](https://gitlab.com/Rickdkk/worklab/tree/master/examples).

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
You need a valid version of Python 3.6+ (because f-strings). This project has a bunch of dependencies for *reasons* so you will also need the scipy ecosystem
, which you probably already have.

### Installing
Option 1: download this package, and run:

```
python setup.py install
```

Option 2: installing with pip is even easier:
```
pip install git+https://gitlab.com/Rickdkk/worklab
```

Option 3: don't install it and just include the scripts in your working directory (why though?).

To verify if everything works simply try to import worklab:

```
python
import worklab as wl
```
That's it.

## Breakdown
* Core files:
	* com.py: 		Provides functions for reading and writing data, use `load` to infer filetype and automatically read
	it. If you use a different naming scheme you can always call the specific load functions.
	* kinetics.py: 	Contains all essentials for measurement wheel and ergometer data. You only need the top-level function 
	`auto_process` for most use-cases.
	* move.py: 	    Contains kinematics and movement related functions, which for now is only the means to resample NGIMU
	data and calculate wheelchair velocity from it. Will include some 3D kinematics functions as soon as I need those.
	* physio.py: 	Contains physiological calculations, which for now is basically nothing as the spirometer does 
	everything for you. Might include EMG and the likes later though.
	* plots.py:     Contains some basic plotting functionalities for plots that become repetitive, needs some TLC to 
	become really useful
	* utils.py:     Contains all functions that are useful for more than one application (e.g. filtering is important
	independent of what instrument you use)

The return of a function is a Pandas DataFrame in 9/10 cases. This means that you can also use all Pandas goodness.

## Examples
You can find some Jupyter Notebook examples [here](https://gitlab.com/Rickdkk/worklab/tree/master/examples).

## Projects using same code
* Viewer (built with PyQt) - source unfortunately was lost when my laptop was stolen	
* [Coast-down analyzer](https://gitlab.com/Rickdkk/coast_down_test) (built with Pyside 2)

## Reporting errors
If you find an error or mistake, which entirely possible, please contact me or submit an issue through this page.

## Authors
* **Rick de Klerk** - *Initial work* - [gitlab](https://gitlab.com/rickdkk) - [UMCG](https://www.rug.nl/staff/r.de.klerk/)

## License
This project is licensed under the GNU GPLv3 - see the [LICENSE](LICENSE) file for details. If you want to refer to this 
package please use this DOI: 10.5281/zenodo.3268671, or cite: R.de Klerk. (2019, July 4). Worklab: a wheelchair biomechanics mini-package (Version 1.0.0). Zenodo. http://doi.org/10.5281/zenodo.3268671

## Acknowledgments
* Thanks to [R.J.K. Vegter](https://www.rug.nl/staff/r.j.k.vegter/) for providing information on the Optipush and SMARTwheel systems.
* Thanks to R.M.A. van der Slikke for providing information on skid correction.
* Thanks to the people at Umaco for answering my questions however dumb they may be.

## References
* Vegter, R. J., Lamoth, C. J., De Groot, S., Veeger, D. H., & Van der Woude, L. H. (2013). Variability in bimanual wheelchair propulsion: consistency of two instrumented wheels during handrim wheelchair propulsion on a motor driven treadmill. Journal of neuroengineering and rehabilitation, 10(1), 9.
* Van der Slikke, R. M. A., Berger, M. A. M., Bregman, D. J. J., & Veeger, H. E. J. (2015). Wheel skid correction is a prerequisite to reliably measure wheelchair sports kinematics based on inertial sensors. Procedia Engineering, 112, 207-212.
