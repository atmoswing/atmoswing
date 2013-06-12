#ifndef ASMETHODCALIBRATOR_H
#define ASMETHODCALIBRATOR_H

#include <asIncludes.h>
#include <asMethodStandard.h>
#include <asParametersCalibration.h>
#include <asResultsAnalogsDates.h>
#include <asCatalogPredictorsArchive.h>
#include <asDataPredictorArchive.h>
#include <asResultsParametersArray.h>
#include <asResultsAnalogsDates.h>
#include <asResultsAnalogsValues.h>
#include <asResultsAnalogsForecastScores.h>
#include <asResultsAnalogsForecastScoreFinal.h>
#include <asResultsAnalogsScoresMap.h>
#include <asParametersCalibration.h>
#include <asPredictorCriteria.h>
#include <asGeoAreaCompositeGrid.h>
#include <asTimeArray.h>
#include <asProcessor.h>
#include <asProcessorForecastScore.h>
#include <asPreprocessor.h>
#include <asForecastScore.h>


class asMethodCalibrator: public asMethodStandard
{
public:
    asMethodCalibrator();
    virtual ~asMethodCalibrator();

    bool GetAnalogsDates(asResultsAnalogsDates &results, asParametersScoring &params, int i_step, bool &containsNaNs);
    bool GetAnalogsSubDates(asResultsAnalogsDates &results, asParametersScoring &params, asResultsAnalogsDates &anaDates, int i_step, bool &containsNaNs);
    bool GetAnalogsValues(asResultsAnalogsValues &results, asParametersScoring &params, asResultsAnalogsDates &anaDates, int i_step);
    bool GetAnalogsForecastScores(asResultsAnalogsForecastScores &results, asParametersScoring &params, asResultsAnalogsValues &anaValues, int i_step);
    bool GetAnalogsForecastScoreFinal(asResultsAnalogsForecastScoreFinal &results, asParametersScoring &params, asResultsAnalogsForecastScores &anaScores, int i_step);
    bool SubProcessAnalogsNumber(asParametersCalibration &params, asResultsAnalogsDates &anaDatesPrevious, int i_step = 0);

    void DeletePreloadedData();
    void ClearAll();
    void ClearTemp();
    void ClearScores();
    void PushBackBestTemp();
    void RemoveNaNsInTemp();
    void KeepBestTemp();
    void PushBackFirstTemp();
    void KeepFirstTemp();
    void SortScoresAndParameters();
    void SortScoresAndParametersTemp();
    bool PushBackInTempIfBetter(asParametersCalibration &params, asResultsAnalogsForecastScoreFinal &scoreFinal);
    bool KeepIfBetter(asParametersCalibration &params, asResultsAnalogsForecastScoreFinal &scoreFinal);
    bool SetSelectedParameters(asResultsParametersArray &results);
    bool SetBestParameters(asResultsParametersArray &results);

    bool Manager();

    virtual bool Validate(const int bestscorerow = 0);

    void SetScore(float valCalib)
    {
        m_ScoresCalibTemp.push_back(valCalib);
    }

    void SetScoreOrder(Order val)
    {
        m_ScoreOrder = val;
    }

    float GetScoreClimatology()
    {
        return m_ScoreClimatology;
    }

    void SetScoreClimatology(float val)
    {
        m_ScoreClimatology = val;
    }

    void SetAlternateCatalogPredictorsArchiveFilePath(const wxString &val)
    {
        m_AlternateCatalogPredictorsArchiveFilePath = val;
    }

    wxString GetAlternateCatalogPredictorsArchiveFilePath()
    {
        return m_AlternateCatalogPredictorsArchiveFilePath;
    }


protected:
    VectorFloat m_ScoresCalib;
    VectorFloat m_ScoresCalibTemp;
    Order m_ScoreOrder;
    float m_ScoreValid;
    float m_ScoreClimatology;
    wxString m_AlternateCatalogPredictorsArchiveFilePath;
    std::vector <asParametersCalibration> m_Parameters;
    std::vector <asParametersCalibration> m_ParametersTemp;
    asParametersCalibration m_OriginalParams;
    bool m_Preloaded;
    bool m_ValidationMode;
    std::vector < std::vector < std::vector < std::vector < asDataPredictorArchive* > > > > m_PreloadedArchive;
    VVectorBool m_PreloadedArchivePointerCopy;

	virtual bool Calibrate(asParametersCalibration &params) = 0;
	bool PreloadData(asParametersScoring &params);

private:


};

#endif // ASMETHODCALIBRATOR_H
