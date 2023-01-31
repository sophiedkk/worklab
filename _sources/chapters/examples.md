# Minimal examples

This page contains some commonly requested examples in Jupyter
notebooks. The general approach to processing the data used in this
package is usually fairly similar:

-   load the data
-   filter the data
-   pre-process the data
-   reduce the data (e.g. by slicing and/or calculating means)

Loading the data can usually be performed by using the
[.com.load]{.title-ref} function. These functions usually don\'t do much
with the data except maybe convert some columns to the SI standard.

Filtering the data is very much up to your own preferences and needs, I
do not know what the contents of your signal are, but there are some
functions available in the package to make this easier.

Pre-processing is largely taken care of. It could be that you need an
extra variable such as, I don\'t know, the third derivative of velocity.
You would need to compute those yourself. If you feel like a variable is
really missing you are welcome to contact me and/or send a pull request
and I will include it.

Reducing the data is very much up to yourself. Given that the data is in
a pandas dataframe this should be straightforward. Pandas dataframes
have excellent support for slicing operations and getting the basic
statistics is as easy as calling the .describe method on said dataframe.

We could go on for days, but it\'s usually best to show an example:
