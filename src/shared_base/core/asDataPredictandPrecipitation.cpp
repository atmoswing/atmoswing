/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL Header Notice in
 * each file and include the License file (licence.txt). If applicable,
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 *
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asDataPredictandPrecipitation.h"

#include "wx/fileconf.h"

#include <asFileNetcdf.h>
#include <asTimeArray.h>
#include <asCatalogPredictands.h>


asDataPredictandPrecipitation::asDataPredictandPrecipitation(DataParameter dataParameter,
                                                             DataTemporalResolution dataTemporalResolution,
                                                             DataSpatialAggregation dataSpatialAggregation)
        : asDataPredictand(dataParameter, dataTemporalResolution, dataSpatialAggregation)
{
    //ctor
    m_hasNormalizedData = true;
    m_hasReferenceValues = true;
    m_returnPeriodNormalization = 10;
    m_isSqrt = false;

    if (dataTemporalResolution == SixHourlyMovingDailyTemporalWindow) {
        m_hasNormalizedData = false;
        m_hasReferenceValues = false;
    }
}

asDataPredictandPrecipitation::~asDataPredictandPrecipitation()
{
    //dtor
}

bool asDataPredictandPrecipitation::InitContainers()
{
    return InitBaseContainers();
}

bool asDataPredictandPrecipitation::Load(const wxString &filePath)
{
    // Open the NetCDF file
    asLogMessage(wxString::Format(_("Opening the file %s"), filePath));
    asFileNetcdf ncFile(filePath, asFileNetcdf::ReadOnly);
    if (!ncFile.Open()) {
        asLogError(wxString::Format(_("Couldn't open file %s"), filePath));
        return false;
    } else {
        asLogMessage(_("File successfully opened"));
    }

    // Load common data
    LoadCommonData(ncFile);

    if (m_dataTemporalResolution != SixHourlyMovingDailyTemporalWindow) {
        // Get global attributes
        m_returnPeriodNormalization = ncFile.GetAttFloat("return_period_normalization");
        m_isSqrt = false;
        if (ncFile.GetAttShort("uses_square_root") == 1) {
            m_isSqrt = true;
        }

        // Get return periods properties
        int returnPeriodsNb = ncFile.GetDimLength("return_periods");
        m_returnPeriods.resize(returnPeriodsNb);
        ncFile.GetVar("return_periods", &m_returnPeriods[0]);
        size_t startReturnPeriodPrecip[2] = {0, 0};
        size_t countReturnPeriodPrecip[2] = {size_t(m_stationsNb), size_t(returnPeriodsNb)};
        m_dailyPrecipitationsForReturnPeriods.resize(m_stationsNb, returnPeriodsNb);
        ncFile.GetVarArray("daily_precipitations_for_return_periods", startReturnPeriodPrecip, countReturnPeriodPrecip,
                           &m_dailyPrecipitationsForReturnPeriods(0, 0));

        // Get normalized data
        size_t IndexStart[2] = {0, 0};
        size_t IndexCount[2] = {size_t(m_timeLength), size_t(m_stationsNb)};
        m_dataNormalized.resize(m_timeLength, m_stationsNb);
        ncFile.GetVarArray("data_normalized", IndexStart, IndexCount, &m_dataNormalized(0, 0));
    }

    // Close the netCDF file
    ncFile.Close();

    return true;
}

bool asDataPredictandPrecipitation::Save(const wxString &AlternateDestinationDir)
{
    // Get the file path
    wxString PredictandDBFilePath = GetDBFilePathSaving(AlternateDestinationDir);

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(PredictandDBFilePath, asFileNetcdf::Replace);
    if (!ncFile.Open())
        return false;

    // Set common definitions
    SetCommonDefinitions(ncFile);

    if (m_dataTemporalResolution != SixHourlyMovingDailyTemporalWindow) {
        // Define specific dimensions.
        ncFile.DefDim("return_periods", (int) m_returnPeriods.size());

        // The dimensions name array is used to pass the dimensions to the variable.
        VectorStdString DimNames2D;
        DimNames2D.push_back("time");
        DimNames2D.push_back("stations");
        VectorStdString DimNameReturnPeriods;
        DimNameReturnPeriods.push_back("return_periods");
        VectorStdString DimNames2DReturnPeriods;
        DimNames2DReturnPeriods.push_back("stations");
        DimNames2DReturnPeriods.push_back("return_periods");

        // Define specific variables
        ncFile.DefVar("data_normalized", NC_FLOAT, 2, DimNames2D);
        ncFile.DefVar("return_periods", NC_FLOAT, 1, DimNameReturnPeriods);
        ncFile.DefVar("daily_precipitations_for_return_periods", NC_FLOAT, 2, DimNames2DReturnPeriods);

        // Put general attributes
        ncFile.PutAtt("return_period_normalization", &m_returnPeriodNormalization);
        short isSqrt = 0;
        if (m_isSqrt) {
            isSqrt = 1;
        }
        ncFile.PutAtt("uses_square_root", &isSqrt);

        // Put attributes for the data variable
        ncFile.PutAtt("long_name", "Normalized data", "data_normalized");
        ncFile.PutAtt("var_desc", "Normalized data", "data_normalized");
        ncFile.PutAtt("units", "-", "data_normalized");

        // Put attributes for the return periods variable
        ncFile.PutAtt("long_name", "Return periods", "return_periods");
        ncFile.PutAtt("var_desc", "Return periods", "return_periods");
        ncFile.PutAtt("units", "year", "return_periods");

        // Put attributes for the daily precipitations corresponding to the return periods
        ncFile.PutAtt("long_name", "Daily precipitation for return periods", "daily_precipitations_for_return_periods");
        ncFile.PutAtt("var_desc", "Daily precipitation corresponding to the return periods",
                      "daily_precipitations_for_return_periods");
        ncFile.PutAtt("units", "mm", "daily_precipitations_for_return_periods");
    }

    // End definitions: leave define mode
    ncFile.EndDef();

    // Save common data
    SaveCommonData(ncFile);

    if (m_dataTemporalResolution != SixHourlyMovingDailyTemporalWindow) {
        // Provide sizes for specific variables
        size_t start2[] = {0, 0};
        size_t count2[] = {size_t(m_timeLength), size_t(m_stationsNb)};
        size_t startReturnPeriod[] = {0};
        size_t countReturnPeriod[] = {size_t(m_returnPeriods.size())};
        size_t startReturnPeriodPrecip[] = {0, 0};
        size_t countReturnPeriodPrecip[] = {size_t(m_stationsNb), size_t(m_returnPeriods.size())};

        // Write specific data
        ncFile.PutVarArray("data_normalized", start2, count2, &m_dataNormalized(0, 0));
        ncFile.PutVarArray("return_periods", startReturnPeriod, countReturnPeriod, &m_returnPeriods(0));
        ncFile.PutVarArray("daily_precipitations_for_return_periods", startReturnPeriodPrecip, countReturnPeriodPrecip,
                           &m_dailyPrecipitationsForReturnPeriods(0, 0));
    }

    // Close:save new netCDF dataset
    ncFile.Close();

    return true;
}

bool asDataPredictandPrecipitation::BuildPredictandDB(const wxString &catalogFilePath, const wxString &AlternateDataDir,
                                                      const wxString &AlternatePatternDir,
                                                      const wxString &AlternateDestinationDir)
{
    if (!g_unitTesting)
        asLogMessage(_("Building the predictand DB."));

    // Initialize the members
    if (!InitMembers(catalogFilePath))
        return false;

    // Resize matrices
    if (!InitContainers())
        return false;

    // Load data from files
    if (!ParseData(catalogFilePath, AlternateDataDir, AlternatePatternDir))
        return false;

    if (m_dataTemporalResolution != SixHourlyMovingDailyTemporalWindow) {
        // Make the Gumbel adjustment
        if (!MakeGumbelAdjustment())
            return false;

        // Process the normalized Precipitation
        if (!BuildDataNormalized())
            return false;

        // Process daily precipitations for all return periods
        if (!BuildDailyPrecipitationsForAllReturnPeriods())
            return false;
    }

    Save(AlternateDestinationDir);

    if (!g_unitTesting)
        asLogMessage(_("Predictand DB saved."));

#if wxUSE_GUI
    if (!g_silentMode) {
        wxMessageBox(_("Predictand DB saved."));
    }
#endif

    return true;
}

bool asDataPredictandPrecipitation::MakeGumbelAdjustment()
{
    // Duration of the Precipitation
    Array1DDouble duration;
    if (m_timeStepDays == 1) {
        duration.resize(7);
        duration << 1, 2, 3, 4, 5, 7, 10;
    } else if (m_timeStepDays <= 1.0 / 24.0) {
        duration.resize(14);
        duration << 1.0 / 24.0, 2.0 / 24.0, 3.0 / 24.0, 4.0 / 24.0, 5.0 / 24.0, 6.0 / 24.0, 12.0 /
                                                                                            24.0, 1, 2, 3, 4, 5, 7, 10;
    } else if (m_timeStepDays <= 2.0 / 24.0) {
        duration.resize(13);
        duration << 2.0 / 24.0, 3.0 / 24.0, 4.0 / 24.0, 5.0 / 24.0, 6.0 / 24.0, 12.0 / 24.0, 1, 2, 3, 4, 5, 7, 10;
    } else if (m_timeStepDays <= 6.0 / 24.0) {
        duration.resize(9);
        duration << 6.0 / 24.0, 12.0 / 24.0, 1, 2, 3, 4, 5, 7, 10;
    } else if (m_timeStepDays <= 12.0 / 24.0) {
        duration.resize(8);
        duration << 12.0 / 24.0, 1, 2, 3, 4, 5, 7, 10;
    } else if (m_timeStepDays == 2) {
        duration.resize(5);
        duration << 2, 4, 6, 8, 10;
    } else if (m_timeStepDays == 3) {
        duration.resize(5);
        duration << 3, 6, 9, 12, 15;
    } else if (m_timeStepDays == 7) {
        duration.resize(4);
        duration << 7, 14, 21, 28;
    } else {
        asLogError(_("The data time steps is not correctly defined."));
        duration.resize(7);
        duration << 1, 2, 3, 4, 5, 7, 10;
    }

    // Preprocess cste
    float b_cst = sqrt(6.0) / g_cst_Pi;

    // Resize containers
    m_gumbelDuration.resize(m_stationsNb, duration.size());
    m_gumbelParamA.resize(m_stationsNb, duration.size());
    m_gumbelParamB.resize(m_stationsNb, duration.size());

#if wxUSE_GUI
    // The progress bar
    asDialogProgressBar ProgressBar(_("Making Gumbel adjustments."), duration.size() - 1);
#endif

    for (float i_duration = 0; i_duration < duration.size(); i_duration++) {
        // Get the annual max
        Array2DFloat annualMax = GetAnnualMax(duration[i_duration]);

#if wxUSE_GUI
        if (!ProgressBar.Update(i_duration)) {
            asLogError(_("The process has been canceled by the user."));
            return false;
        }
#endif

        for (int i_st = 0; i_st < m_stationsNb; i_st++) {
            Array1DFloat currentAnnualMax = annualMax.row(i_st);
            int arrayEnd = currentAnnualMax.size() - 1;

            // Check the length of the data
            int dataLength = asTools::CountNotNaN(&currentAnnualMax(0), &currentAnnualMax(arrayEnd));
            if (dataLength < 20) {
                asLogError(
                        _("Caution, a time serie is shorter than 20 years. It is too short to process a Gumbel adjustment."));
                return false;
            } else if (dataLength < 30) {
                asLogWarning(
                        _("Caution, a time serie is shorter than 30 years. It is a bit short to process a Gumbel adjustment."));
            }

            if (!asTools::SortArray(&currentAnnualMax(0), &currentAnnualMax(arrayEnd), Asc))
                return false;
            float mean = asTools::Mean(&currentAnnualMax(0), &currentAnnualMax(arrayEnd));
            float stdev = asTools::StDev(&currentAnnualMax(0), &currentAnnualMax(arrayEnd), asSAMPLE);

            float b = b_cst * stdev;
            float a = mean - b * g_cst_Euler; // EUCON: Euler-Mascheroni constant in math.h

            m_gumbelDuration(i_st, i_duration) = duration[i_duration];
            m_gumbelParamA(i_st, i_duration) = a;
            m_gumbelParamB(i_st, i_duration) = b;
        }
    }
#if wxUSE_GUI
    ProgressBar.Destroy();
#endif

    return true;
}

float asDataPredictandPrecipitation::GetPrecipitationOfReturnPeriod(int i_station, double duration, float returnPeriod)
{
    float F = 1 - (1 / returnPeriod); // Probability of not overtaking
    float u = -log(-log(F)); // Gumbel variable
    Array1DFloat durations = m_gumbelDuration.row(i_station);
    int i_duration = asTools::SortedArraySearch(&durations(0), &durations(durations.size() - 1), duration, 0.00001f);
    return m_gumbelParamB(i_station, i_duration) * u + m_gumbelParamA(i_station, i_duration);
}

bool asDataPredictandPrecipitation::BuildDailyPrecipitationsForAllReturnPeriods()
{
    float duration = 1; // day
    m_returnPeriods.resize(10);
    m_returnPeriods << 2, 2.33f, 5, 10, 20, 50, 100, 200, 300, 500;
    m_dailyPrecipitationsForReturnPeriods.resize(m_stationsNb, m_returnPeriods.size());

    for (int i_station = 0; i_station < m_stationsNb; i_station++) {
        for (int i_retperiod = 0; i_retperiod < m_returnPeriods.size(); i_retperiod++) {
            float F = 1 - (1 / m_returnPeriods[i_retperiod]); // Probability of not overtaking
            float u = -log(-log(F)); // Gumbel variable
            int i_duration = asTools::SortedArraySearch(&m_gumbelDuration(i_station, 0),
                                                        &m_gumbelDuration(i_station, m_gumbelDuration.cols() - 1),
                                                        duration, 0.00001f);
            float val = m_gumbelParamB(i_station, i_duration) * u + m_gumbelParamA(i_station, i_duration);
            wxASSERT(val > 0);
            wxASSERT(val < 500);
            m_dailyPrecipitationsForReturnPeriods(i_station, i_retperiod) = val;
        }
    }

    return true;
}

bool asDataPredictandPrecipitation::BuildDataNormalized()
{
    for (int i_st = 0; i_st < m_stationsNb; i_st++) {
        float Prt = 1.0;
        if (m_returnPeriodNormalization != 0) {
            Prt = GetPrecipitationOfReturnPeriod(i_st, 1, m_returnPeriodNormalization);
        }

        for (int i_time = 0; i_time < m_timeLength; i_time++) {
            if (m_isSqrt) {
                m_dataNormalized(i_time, i_st) = sqrt(m_dataGross(i_time, i_st) / Prt);
            } else {
                m_dataNormalized(i_time, i_st) = m_dataGross(i_time, i_st) / Prt;
            }
        }
    }
    return true;
}
