#include "include_tests.h"
#include "asParametersOptimization.h"

#include "UnitTest++.h"

namespace
{

TEST(ParametersOptimizationLoadFromFile)
{
	wxString str("Testing optimization parameters...\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/parameters_optimization.xml");

    asParametersOptimization params;
    params.LoadFromFile(filepath);

    CHECK_EQUAL(asTime::GetMJD(1962,1,1), params.GetArchiveStart());
    CHECK_EQUAL(asTime::GetMJD(2008,12,31), params.GetArchiveEnd());
    CHECK_EQUAL(asTime::GetMJD(1970,1,1), params.GetCalibrationStart());
    CHECK_EQUAL(asTime::GetMJD(2000,12,31), params.GetCalibrationEnd());
    CHECK_EQUAL(24, params.GetTimeArrayAnalogsTimeStepHours());
    CHECK_EQUAL(24, params.GetTimeArrayTargetTimeStepHours());
    CHECK_EQUAL(1, params.GetTimeArrayAnalogsIntervalDaysIteration());
    CHECK_EQUAL(10, params.GetTimeArrayAnalogsIntervalDaysLowerLimit());
    CHECK_EQUAL(182, params.GetTimeArrayAnalogsIntervalDaysUpperLimit());
    CHECK_EQUAL(false, params.IsTimeArrayAnalogsIntervalDaysLocked());
    CHECK_EQUAL(60, params.GetTimeArrayAnalogsExcludeDays());
    CHECK_EQUAL(true, params.GetTimeArrayAnalogsMode().IsSameAs("DaysInterval"));
    CHECK_EQUAL(true, params.GetTimeArrayTargetMode().IsSameAs("Simple"));

    CHECK_EQUAL(1, params.GetAnalogsNumberIteration(0));
    CHECK_EQUAL(5, params.GetAnalogsNumberLowerLimit(0));
    CHECK_EQUAL(200, params.GetAnalogsNumberUpperLimit(0));
    CHECK_EQUAL(false, params.IsAnalogsNumberLocked(0));

    CHECK_EQUAL(false, params.NeedsPreprocessing(0,0));
    CHECK_EQUAL(true, params.GetPredictorDatasetId(0,0).IsSameAs("NCEP_R-1"));
    CHECK_EQUAL(true, params.GetPredictorDataId(0,0).IsSameAs("hgt"));
    CHECK_EQUAL(500, params.GetPredictorLevel(0,0));
    CHECK_EQUAL(6, params.GetPredictorTimeHoursIteration(0,0));
    CHECK_EQUAL(-48, params.GetPredictorTimeHoursLowerLimit(0,0));
    CHECK_EQUAL(48, params.GetPredictorTimeHoursUpperLimit(0,0));
    CHECK_EQUAL(false, params.IsPredictorTimeHoursLocked(0,0));
    CHECK_EQUAL(true, params.GetPredictorGridType(0,0).IsSameAs("Regular"));
    CHECK_EQUAL(2.5, params.GetPredictorXminIteration(0,0));
    CHECK_EQUAL(300, params.GetPredictorXminLowerLimit(0,0));
    CHECK_EQUAL(450, params.GetPredictorXminUpperLimit(0,0));
    CHECK_EQUAL(false, params.IsPredictorXminLocked(0,0));
    CHECK_EQUAL(1, params.GetPredictorXptsnbIteration(0,0));
    CHECK_EQUAL(1, params.GetPredictorXptsnbLowerLimit(0,0));
    CHECK_EQUAL(21, params.GetPredictorXptsnbUpperLimit(0,0));
    CHECK_EQUAL(false, params.IsPredictorXptsnbLocked(0,0));
    CHECK_EQUAL(2.5, params.GetPredictorXstep(0,0));
    CHECK_EQUAL(2.5, params.GetPredictorYminIteration(0,0));
    CHECK_EQUAL(0, params.GetPredictorYminLowerLimit(0,0));
    CHECK_EQUAL(70, params.GetPredictorYminUpperLimit(0,0));
    CHECK_EQUAL(false, params.IsPredictorYminLocked(0,0));
    CHECK_EQUAL(1, params.GetPredictorYptsnbIteration(0,0));
    CHECK_EQUAL(1, params.GetPredictorYptsnbLowerLimit(0,0));
    CHECK_EQUAL(13, params.GetPredictorYptsnbUpperLimit(0,0));
    CHECK_EQUAL(false, params.IsPredictorYptsnbLocked(0,0));
    CHECK_EQUAL(2.5, params.GetPredictorYstep(0,0));
    CHECK_EQUAL(true, params.GetPredictorCriteria(0,0).IsSameAs("S1"));
    CHECK_CLOSE(0.01, params.GetPredictorWeightIteration(0,0), 0.0001);
    CHECK_CLOSE(0, params.GetPredictorWeightLowerLimit(0,0), 0.0001);
    CHECK_CLOSE(1, params.GetPredictorWeightUpperLimit(0,0), 0.0001);
    CHECK_EQUAL(false, params.IsPredictorWeightLocked(0,0));

    CHECK_EQUAL(true, params.NeedsPreprocessing(0,1));
    CHECK_EQUAL(true, params.GetPreprocessMethod(0,1).IsSameAs("Gradients"));
    CHECK_EQUAL(true, params.GetPreprocessDatasetId(0,1,0).IsSameAs("NCEP_R-1"));
    CHECK_EQUAL(true, params.GetPreprocessDataId(0,1,0).IsSameAs("hgt"));
    CHECK_EQUAL(1000, params.GetPreprocessLevel(0,1,0));
    CHECK_EQUAL(12, params.GetPreprocessTimeHours(0,1,0));
    CHECK_EQUAL(true, params.GetPredictorGridType(0,1).IsSameAs("Regular"));
    CHECK_EQUAL(2.5, params.GetPredictorXminIteration(0,1));
    CHECK_EQUAL(300, params.GetPredictorXminLowerLimit(0,1));
    CHECK_EQUAL(450, params.GetPredictorXminUpperLimit(0,1));
    CHECK_EQUAL(false, params.IsPredictorXminLocked(0,1));
    CHECK_EQUAL(1, params.GetPredictorXptsnbIteration(0,1));
    CHECK_EQUAL(3, params.GetPredictorXptsnbLowerLimit(0,1));
    CHECK_EQUAL(19, params.GetPredictorXptsnbUpperLimit(0,1));
    CHECK_EQUAL(false, params.IsPredictorXptsnbLocked(0,1));
    CHECK_EQUAL(2.5, params.GetPredictorXstep(0,1));
    CHECK_EQUAL(2.5, params.GetPredictorYminIteration(0,1));
    CHECK_EQUAL(0, params.GetPredictorYminLowerLimit(0,1));
    CHECK_EQUAL(70, params.GetPredictorYminUpperLimit(0,1));
    CHECK_EQUAL(false, params.IsPredictorYminLocked(0,1));
    CHECK_EQUAL(1, params.GetPredictorYptsnbIteration(0,1));
    CHECK_EQUAL(1, params.GetPredictorYptsnbLowerLimit(0,1));
    CHECK_EQUAL(9, params.GetPredictorYptsnbUpperLimit(0,1));
    CHECK_EQUAL(false, params.IsPredictorYptsnbLocked(0,1));
    CHECK_EQUAL(2.5, params.GetPredictorYstep(0,1));
    CHECK_EQUAL(true, params.GetPredictorCriteria(0,1).IsSameAs("S1grads"));
    CHECK_CLOSE(0.01, params.GetPredictorWeightIteration(0,1), 0.0001);
    CHECK_CLOSE(0, params.GetPredictorWeightLowerLimit(0,1), 0.0001);
    CHECK_CLOSE(1, params.GetPredictorWeightUpperLimit(0,1), 0.0001);
    CHECK_EQUAL(false, params.IsPredictorWeightLocked(0,1));

    CHECK_EQUAL(40, params.GetPredictandStationIds()[0]);

    CHECK_EQUAL(true, params.GetForecastScoreName().IsSameAs("CRPSAR"));

    CHECK_EQUAL(true, params.ForecastScoreNeedsPostprocessing());
    CHECK_EQUAL(true, params.GetForecastScorePostprocessMethod().IsSameAs("DuplicationOnCriteriaExponent"));
    CHECK_CLOSE(2.1, params.GetForecastScorePostprocessDupliExp(), 0.0001);

    CHECK_EQUAL(true, params.GetForecastScoreTimeArrayMode().IsSameAs("Simple"));
    CHECK_CLOSE(357434, params.GetForecastScoreTimeArrayDate(), 0.0001);
    CHECK_CLOSE(60, params.GetForecastScoreTimeArrayIntervalDays(), 0.0001);
}

TEST(ParametersOptimizationLoadFromFileAndInitRandomValues)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/parameters_optimization.xml");

    asParametersOptimization params;
    params.LoadFromFile(filepath);

    params.InitRandomValues();

    CHECK_EQUAL(asTime::GetMJD(1962,1,1), params.GetArchiveStart());
    CHECK_EQUAL(asTime::GetMJD(2008,12,31), params.GetArchiveEnd());
    CHECK_EQUAL(asTime::GetMJD(1970,1,1), params.GetCalibrationStart());
    CHECK_EQUAL(asTime::GetMJD(2000,12,31), params.GetCalibrationEnd());
    CHECK_EQUAL(24, params.GetTimeArrayAnalogsTimeStepHours());
    CHECK_EQUAL(24, params.GetTimeArrayTargetTimeStepHours());
    CHECK_EQUAL(1, params.GetTimeArrayAnalogsIntervalDaysIteration());
    CHECK_EQUAL(10, params.GetTimeArrayAnalogsIntervalDaysLowerLimit());
    CHECK_EQUAL(182, params.GetTimeArrayAnalogsIntervalDaysUpperLimit());
    CHECK_EQUAL(false, params.IsTimeArrayAnalogsIntervalDaysLocked());
    CHECK_EQUAL(60, params.GetTimeArrayAnalogsExcludeDays());
    CHECK_EQUAL(true, params.GetTimeArrayAnalogsMode().IsSameAs("DaysInterval"));
    CHECK_EQUAL(true, params.GetTimeArrayTargetMode().IsSameAs("Simple"));

    CHECK_EQUAL(1, params.GetAnalogsNumberIteration(0));
    CHECK_EQUAL(5, params.GetAnalogsNumberLowerLimit(0));
    CHECK_EQUAL(200, params.GetAnalogsNumberUpperLimit(0));
    CHECK_EQUAL(false, params.IsAnalogsNumberLocked(0));

    CHECK_EQUAL(false, params.NeedsPreprocessing(0,0));
    CHECK_EQUAL(true, params.GetPredictorDatasetId(0,0).IsSameAs("NCEP_R-1"));
    CHECK_EQUAL(true, params.GetPredictorDataId(0,0).IsSameAs("hgt"));
    CHECK_EQUAL(500, params.GetPredictorLevel(0,0));
    CHECK_EQUAL(6, params.GetPredictorTimeHoursIteration(0,0));
    CHECK_EQUAL(-48, params.GetPredictorTimeHoursLowerLimit(0,0));
    CHECK_EQUAL(48, params.GetPredictorTimeHoursUpperLimit(0,0));
    CHECK_EQUAL(false, params.IsPredictorTimeHoursLocked(0,0));
    CHECK_EQUAL(true, params.GetPredictorGridType(0,0).IsSameAs("Regular"));
    CHECK_EQUAL(2.5, params.GetPredictorXminIteration(0,0));
    CHECK_EQUAL(300, params.GetPredictorXminLowerLimit(0,0));
    CHECK_EQUAL(450, params.GetPredictorXminUpperLimit(0,0));
    CHECK_EQUAL(false, params.IsPredictorXminLocked(0,0));
    CHECK_EQUAL(1, params.GetPredictorXptsnbIteration(0,0));
    CHECK_EQUAL(1, params.GetPredictorXptsnbLowerLimit(0,0));
    CHECK_EQUAL(21, params.GetPredictorXptsnbUpperLimit(0,0));
    CHECK_EQUAL(false, params.IsPredictorXptsnbLocked(0,0));
    CHECK_EQUAL(2.5, params.GetPredictorXstep(0,0));
    CHECK_EQUAL(2.5, params.GetPredictorYminIteration(0,0));
    CHECK_EQUAL(0, params.GetPredictorYminLowerLimit(0,0));
    CHECK_EQUAL(70, params.GetPredictorYminUpperLimit(0,0));
    CHECK_EQUAL(false, params.IsPredictorYminLocked(0,0));
    CHECK_EQUAL(1, params.GetPredictorYptsnbIteration(0,0));
    CHECK_EQUAL(1, params.GetPredictorYptsnbLowerLimit(0,0));
    CHECK_EQUAL(13, params.GetPredictorYptsnbUpperLimit(0,0));
    CHECK_EQUAL(false, params.IsPredictorYptsnbLocked(0,0));
    CHECK_EQUAL(2.5, params.GetPredictorYstep(0,0));
    CHECK_EQUAL(true, params.GetPredictorCriteria(0,0).IsSameAs("S1"));
    CHECK_CLOSE(0.01, params.GetPredictorWeightIteration(0,0), 0.0001);
    CHECK_CLOSE(0, params.GetPredictorWeightLowerLimit(0,0), 0.0001);
    CHECK_CLOSE(1, params.GetPredictorWeightUpperLimit(0,0), 0.0001);
    CHECK_EQUAL(false, params.IsPredictorWeightLocked(0,0));

    CHECK_EQUAL(true, params.NeedsPreprocessing(0,1));
    CHECK_EQUAL(true, params.GetPreprocessMethod(0,1).IsSameAs("Gradients"));
    CHECK_EQUAL(true, params.GetPreprocessDatasetId(0,1,0).IsSameAs("NCEP_R-1"));
    CHECK_EQUAL(true, params.GetPreprocessDataId(0,1,0).IsSameAs("hgt"));
    CHECK_EQUAL(1000, params.GetPreprocessLevel(0,1,0));
    CHECK_EQUAL(12, params.GetPreprocessTimeHours(0,1,0));
    CHECK_EQUAL(true, params.GetPredictorGridType(0,1).IsSameAs("Regular"));
    CHECK_EQUAL(2.5, params.GetPredictorXminIteration(0,1));
    CHECK_EQUAL(300, params.GetPredictorXminLowerLimit(0,1));
    CHECK_EQUAL(450, params.GetPredictorXminUpperLimit(0,1));
    CHECK_EQUAL(false, params.IsPredictorXminLocked(0,1));
    CHECK_EQUAL(1, params.GetPredictorXptsnbIteration(0,1));
    CHECK_EQUAL(3, params.GetPredictorXptsnbLowerLimit(0,1));
    CHECK_EQUAL(19, params.GetPredictorXptsnbUpperLimit(0,1));
    CHECK_EQUAL(false, params.IsPredictorXptsnbLocked(0,1));
    CHECK_EQUAL(2.5, params.GetPredictorXstep(0,1));
    CHECK_EQUAL(2.5, params.GetPredictorYminIteration(0,1));
    CHECK_EQUAL(0, params.GetPredictorYminLowerLimit(0,1));
    CHECK_EQUAL(70, params.GetPredictorYminUpperLimit(0,1));
    CHECK_EQUAL(false, params.IsPredictorYminLocked(0,1));
    CHECK_EQUAL(1, params.GetPredictorYptsnbIteration(0,1));
    CHECK_EQUAL(1, params.GetPredictorYptsnbLowerLimit(0,1));
    CHECK_EQUAL(9, params.GetPredictorYptsnbUpperLimit(0,1));
    CHECK_EQUAL(false, params.IsPredictorYptsnbLocked(0,1));
    CHECK_EQUAL(2.5, params.GetPredictorYstep(0,1));
    CHECK_EQUAL(true, params.GetPredictorCriteria(0,1).IsSameAs("S1grads"));
    CHECK_CLOSE(0.01, params.GetPredictorWeightIteration(0,1), 0.0001);
    CHECK_CLOSE(0, params.GetPredictorWeightLowerLimit(0,1), 0.0001);
    CHECK_CLOSE(1, params.GetPredictorWeightUpperLimit(0,1), 0.0001);
    CHECK_EQUAL(false, params.IsPredictorWeightLocked(0,1));

    CHECK_EQUAL(40, params.GetPredictandStationIds()[0]);

    CHECK_EQUAL(true, params.GetForecastScoreName().IsSameAs("CRPSAR"));

    CHECK_EQUAL(true, params.ForecastScoreNeedsPostprocessing());
    CHECK_EQUAL(true, params.GetForecastScorePostprocessMethod().IsSameAs("DuplicationOnCriteriaExponent"));
    CHECK_CLOSE(2.1, params.GetForecastScorePostprocessDupliExp(), 0.0001);

    CHECK_EQUAL(true, params.GetForecastScoreTimeArrayMode().IsSameAs("Simple"));
    CHECK_CLOSE(357434, params.GetForecastScoreTimeArrayDate(), 0.0001);
    CHECK_CLOSE(60, params.GetForecastScoreTimeArrayIntervalDays(), 0.0001);
}

}
