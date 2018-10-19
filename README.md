# Worklab data analysis package

Essential data analysis scripts used in my project researching the [Lode Esseda](https://www.lode.nl/en/product/esseda-wheelchair-ergometer/637) wheelchair ergometer in the worklab. Includes all basic io and calculations for all equipment in the worklab.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need a valid version of Python 3.x. This project has a bunch of dependencies for *reasons* so you will also need: matplotlib, numpy, pandas, and scipy.

### Installing

Install Python on your system and run:

```
python setup.py install
```

Download or pull all code from this page and to verify if everything works run:

```
python
import worklab
```

## Breakdown

* Core files:
	* data_io.py: 		Provides functions for reading and writing data, use io.load() to infer filetype
	* data_calcs.py: 	Contains all essential conversions, calculations, and filtering
	* data_plots.py: 	Some basic plotting functionalities that can be used during testing
	* data_formats.py: 	Overview of data structures that are used; mainly for being explicit.

## Projects using same code

* Viewer (built with PyQt)	

## Reporting errors

If you find an error or mistake, which is likely, please contact me or submit an issue through this page.

## Authors

* **Rick de Klerk** - *Initial work* - [gitlab](https://gitlab.com/rickdkk) - [UMCG](https://www.rug.nl/staff/r.de.klerk/)

## License

This project is licensed under the GNU GPLv3 - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Thanks to [R.J.K. Vegter](https://www.rug.nl/staff/r.j.k.vegter/) for providing information on the Optipush and SMARTwheel systems
* Thanks to the people at Umaco for answering my questions however dumb they may be
