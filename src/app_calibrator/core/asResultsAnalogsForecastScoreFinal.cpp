#include "asResultsAnalogsForecastScoreFinal.h"

#include "asFileNetcdf.h"
#include "asParametersScoring.h"


asResultsAnalogsForecastScoreFinal::asResultsAnalogsForecastScoreFinal()
:
asResults()
{
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Read("/Calibration/IntermediateResults/SaveFinalForecastScore", &m_SaveIntermediateResults, false);
    wxFileConfig::Get()->Read("/Calibration/IntermediateResults/LoadFinalForecastScore", &m_LoadIntermediateResults, false);
    ThreadsManager().CritSectionConfig().Leave();
}

asResultsAnalogsForecastScoreFinal::~asResultsAnalogsForecastScoreFinal()
{
    //dtor
}

void asResultsAnalogsForecastScoreFinal::Init(asParametersScoring &params)
{
    if(m_SaveIntermediateResults || m_LoadIntermediateResults) BuildFileName(params);

    // Set to nan to avoid keeping old results
    m_ForecastScore = NaNFloat;
}

void asResultsAnalogsForecastScoreFinal::BuildFileName(asParametersScoring &params)
{
    ThreadsManager().CritSectionConfig().Enter();
    m_FilePath = wxFileConfig::Get()->Read("/StandardPaths/IntermediateResultsDir", asConfig::GetDefaultUserWorkingDir() + "IntermediateResults" + DS);
    ThreadsManager().CritSectionConfig().Leave();
    m_FilePath.Append(DS);
    m_FilePath.Append(wxString::Format("AnalogsForecastScoreFinal_id%d_step%d", params.GetPredictandStationId(), m_CurrentStep));
    m_FilePath.Append(".nc");
}

bool asResultsAnalogsForecastScoreFinal::Save(const wxString &AlternateFilePath)
{
    // If we don't want to save, skip
    if(!m_SaveIntermediateResults) return false;
    wxString message = _("Saving intermediate file: ") + m_FilePath;
    asLogMessage(message);

    // Get the file path
    wxString ResultsFile;
    if (AlternateFilePath.IsEmpty())
    {
        ResultsFile = m_FilePath;
    }
    else
    {
        ResultsFile = AlternateFilePath;
    }

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(ResultsFile, asFileNetcdf::Replace);
    if(!ncFile.Open())
    {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Define dimensions.
    ncFile.DefDim("forecastscore");

    // The dimensions name array is used to pass the dimensions to the variable.
    VectorStdString DimNames1;
    DimNames1.push_back("forecastscore");

    // Define variables
    ncFile.DefVar("forecastscore", NC_FLOAT, 1, DimNames1);

    // Put attributes
    DefForecastScoreFinalAttributes(ncFile);

    // End definitions: leave define mode
    ncFile.EndDef();

    // Provide sizes for variables
    size_t start1D[] = {0};
    size_t count1D[] = {1};

    // Write data
    ncFile.PutVarArray("forecastscore", start1D, count1D, &m_ForecastScore);

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}

bool asResultsAnalogsForecastScoreFinal::Load(const wxString &AlternateFilePath)
{
    // Makes no sense to load at this stage.
    return false;
}
