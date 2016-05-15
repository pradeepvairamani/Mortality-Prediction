This Directory already contains the cleaned data files. For details on how to generate the data files see the final section.

RUNNING THE FILE:
1) Navigate to the "code" directory.
2) Make sure that the "SVMfile" and "SVMcombined" are not avilable in the code directory (If they are there delete them before running the code)
3) Execute the code using the command `sbt/sbt compile run`
4) To run the various machine learning models, execute "python models.py" after the SVMlite folders have been created.


GENERATING THE DATA FILES:
1) Navigate to the "code/sql_code" directory
2) For generating the notes view run the command "psql -h sunlab.cncuaxfkq7hp.us-east-1.rds.amazonaws.com --port=5432 --username=<username> --password --dbname=mimic3 -f intermediatenotes.sql"
3) For generating the baseline features, run the command "psql -h sunlab.cncuaxfkq7hp.us-east-1.rds.amazonaws.com --port=5432 --username=pvairamani3 --password --dbname=mimic3 -f intermediate_view.sql"
4) Execute the sql queries in "https://github.com/MIT-LCP/mimic-code/tree/master/severityscores" to build the views of the various severeity scores.
5) To get the final baseline_view, execute the command "psql -h sunlab.cncuaxfkq7hp.us-east-1.rds.amazonaws.com --port=5432 --username=pvairamani3 --password --dbname=mimic3 -f baseline_view.sql"
6) To copy the contents of the view into a csv enter the psql shell using the command "psql -h sunlab.cncuaxfkq7hp.us-east-1.rds.amazonaws.com --port=5432 --username=<username> --password --dbname=mimic3;"
7) Copy the contents using the command "\copy (Select (*) From intermediatenotes_pvairamani3) To '<path_to_output_csv>/nostopwords.csv' With CSV;"
8) Copy the contents using the command "\copy (Select (*) From baselinefeatures_pvairamani3) To '<path_to_output_csv>' With CSV;"
9) Move these csvs to the "code/data" folder
10) Run "python remove_onix.py" in the data folder to create a new csv without the Onix stopwords.