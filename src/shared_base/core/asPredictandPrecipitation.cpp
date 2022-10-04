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

#include "asPredictandPrecipitation.h"

#include "asCatalogPredictands.h"
#include "asFileNetcdf.h"
#include "asTimeArray.h"

asPredictandPrecipitation::asPredictandPrecipitation(Parameter dataParameter, TemporalResolution dataTemporalResolution,
                                                     SpatialAggregation dataSpatialAggregation)
    : asPredictand(dataParameter, dataTemporalResolution, dataSpatialAggregation),
      m_returnPeriodNormalization(10),
      m_isSqrt(false) {
    m_hasNormalizedData = true;
    m_hasReferenceValues = true;

    if (dataTemporalResolution == OneHourlyMTW || dataTemporalResolution == ThreeHourlyMTW ||
        dataTemporalResolution == SixHourlyMTW || dataTemporalResolution == TwelveHourlyMTW) {
        m_hasNormalizedData = false;
        m_hasReferenceValues = false;
    }
}

bool asPredictandPrecipitation::InitContainers() {
    return InitBaseContainers();
}

bool asPredictandPrecipitation::Load(const wxString &filePath) {
    // Open the NetCDF file
    wxLogVerbose(_("Opening the file %s"), filePath);
    asFileNetcdf ncFile(filePath, asFileNetcdf::ReadOnly);
    if (!ncFile.Open()) {
        wxLogError(_("Couldn't open file %s"), filePath);
        return false;
    } else {
        wxLogVerbose(_("File successfully opened"));
    }

    // Load common data
    LoadCommonData(ncFile);

    if (m_hasNormalizedData) {
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
        size_t indexStart[2] = {0, 0};
        size_t indexCount[2] = {size_t(m_timeLength), size_t(m_stationsNb)};
        m_dataNormalized.resize(m_timeLength, m_stationsNb);
        ncFile.GetVarArray("data_normalized", indexStart, indexCount, &m_dataNormalized(0, 0));
    }

    // Close the netCDF file
    ncFile.Close();

    return true;
}

bool asPredictandPrecipitation::Save(const wxString &destinationDir) const {
    // Get the file path
    wxString predictandDBFilePath = GetDBFilePathSaving(destinationDir);

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(predictandDBFilePath, asFileNetcdf::Replace);
    if (!ncFile.Open()) return false;

    // Set common definitions
    SetCommonDefinitions(ncFile);

    if (m_hasNormalizedData) {
        // Define specific dimensions.
        ncFile.DefDim("return_periods", (int)m_returnPeriods.size());

        // The dimensions name array is used to pass the dimensions to the variable.
        vstds dimNames2D;
        dimNames2D.push_back("time");
        dimNames2D.push_back("stations");
        vstds dimNameReturnPeriods;
        dimNameReturnPeriods.push_back("return_periods");
        vstds dimNames2DReturnPeriods;
        dimNames2DReturnPeriods.push_back("stations");
        dimNames2DReturnPeriods.push_back("return_periods");

        // Define specific variables
        ncFile.DefVar("data_normalized", NC_FLOAT, 2, dimNames2D);
        ncFile.DefVar("return_periods", NC_FLOAT, 1, dimNameReturnPeriods);
        ncFile.DefVar("daily_precipitations_for_return_periods", NC_FLOAT, 2, dimNames2DReturnPeriods);

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

    if (m_hasNormalizedData) {
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

bool asPredictandPrecipitation::BuildPredictandDB(const wxString &catalogFilePath, const wxString &dataDir,
                                                  const wxString &patternDir, const wxString &destinationDir) {

    // Initialize the members
    if (!InitMembers(catalogFilePath)) return false;

    // Resize matrices
    if (!InitContainers()) return false;

    // Load data from files
    if (!ParseData(catalogFilePath, dataDir, patternDir)) return false;

    if (m_hasNormalizedData) {
        // Make the Gumbel adjustment
        if (!MakeGumbelAdjustment()) return false;

        // Process the normalized Precipitation
        if (!BuildDataNormalized()) return false;

        // Process daily precipitations for all return periods
        if (!BuildDailyPrecipitationsForAllReturnPeriods()) return false;
    }

    if (!destinationDir.IsEmpty()) {
        if (!Save(destinationDir)) return false;
    }

    wxLogVerbose(_("Predictand DB saved."));

#if USE_GUI
    if (!g_silentMode) {
        wxMessageBox(_("Predictand DB saved."));
    }
#endif

    return true;
}

bool asPredictandPrecipitation::MakeGumbelAdjustment() {
    // Duration of the Precipitation
    a1d duration;
    if (m_timeStepDays == 1) {
        duration.resize(7);
        duration << 1, 2, 3, 4, 5, 7, 10;
    } else if (m_timeStepDays <= 1.0 / 24.0) {
        duration.resize(14);
        duration << 1.0 / 24.0, 2.0 / 24.0, 3.0 / 24.0, 4.0 / 24.0, 5.0 / 24.0, 6.0 / 24.0, 12.0 / 24.0, 1, 2, 3, 4, 5,
            7, 10;
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
        wxLogError(_("The data time steps is not correctly defined."));
        duration.resize(7);
        duration << 1, 2, 3, 4, 5, 7, 10;
    }

    // Preprocess cste
    float b_cst = std::sqrt(6.0) / g_cst_Pi;

    // Resize containers
    m_gumbelDuration.resize(m_stationsNb, duration.size());
    m_gumbelParamA.resize(m_stationsNb, duration.size());
    m_gumbelParamB.resize(m_stationsNb, duration.size());

#if USE_GUI
    // The progress bar
    asDialogProgressBar ProgressBar(_("Making Gumbel adjustments."), duration.size() - 1);
#endif

    for (int iDuration = 0; iDuration < duration.size(); iDuration++) {
        // Get the annual max
        a2f annualMax = GetAnnualMax(duration[iDuration]);

#if USE_GUI
        if (!ProgressBar.Update(iDuration)) {
            wxLogError(_("The process has been canceled by the user."));
            return false;
        }
#endif

        for (int iStat = 0; iStat < m_stationsNb; iStat++) {
            a1f currentAnnualMax = annualMax.row(iStat);
            int arrayEnd = currentAnnualMax.size() - 1;

            // Check the length of the data
            int dataLength = asCountNotNaN(&currentAnnualMax(0), &currentAnnualMax(arrayEnd));
            if (dataLength < 20) {
                wxLogError(_(
                    "Caution, a time serie is shorter than 20 years. It is too short to process a Gumbel adjustment."));
                return false;
            } else if (dataLength < 30) {
                wxLogWarning(
                    _("Caution, a time serie is shorter than 30 years. It is a bit short to process a Gumbel "
                      "adjustment."));
            }

            if (!asSortArray(&currentAnnualMax(0), &currentAnnualMax(arrayEnd), Asc)) return false;
            float mean = asMean(&currentAnnualMax(0), &currentAnnualMax(arrayEnd));
            float stdev = asStDev(&currentAnnualMax(0), &currentAnnualMax(arrayEnd), asSAMPLE);

            float b = b_cst * stdev;
            float a = mean - b * g_cst_Euler;  // EUCON: Euler-Mascheroni constant in math.h

            m_gumbelDuration(iStat, iDuration) = duration[iDuration];
            m_gumbelParamA(iStat, iDuration) = a;
            m_gumbelParamB(iStat, iDuration) = b;
        }
    }
#if USE_GUI
    ProgressBar.Destroy();
#endif

    return true;
}

float asPredictandPrecipitation::GetPrecipitationOfReturnPeriod(int iStat, double duration, float returnPeriod) const {
    float F = 1 - (1 / returnPeriod);  // Probability of not overtaking
    float u = -log(-log(F));           // Gumbel variable
    a1f durations = m_gumbelDuration.row(iStat);
    int iDuration = asFind(&durations(0), &durations(durations.size() - 1), duration, 0.00001f);
    return m_gumbelParamB(iStat, iDuration) * u + m_gumbelParamA(iStat, iDuration);
}

bool asPredictandPrecipitation::BuildDailyPrecipitationsForAllReturnPeriods() {
    float duration = 1;  // day
    m_returnPeriods.resize(10);
    m_returnPeriods << 2, 2.33f, 5, 10, 20, 50, 100, 200, 300, 500;
    m_dailyPrecipitationsForReturnPeriods.resize(m_stationsNb, m_returnPeriods.size());

    for (int iStat = 0; iStat < m_stationsNb; iStat++) {
        for (int iRetPeriod = 0; iRetPeriod < m_returnPeriods.size(); iRetPeriod++) {
            float F = 1 - (1 / m_returnPeriods[iRetPeriod]);  // Probability of not overtaking
            float u = -log(-log(F));  // Gumbel variable
            int iDuration = asFind(&m_gumbelDuration(iStat, 0), &m_gumbelDuration(iStat, m_gumbelDuration.cols() - 1),
                                   duration, 0.00001f);
            float val = m_gumbelParamB(iStat, iDuration) * u + m_gumbelParamA(iStat, iDuration);
            wxASSERT(val > 0);
            wxASSERT(val < 1000);
            m_dailyPrecipitationsForReturnPeriods(iStat, iRetPeriod) = val;
        }
    }

    return true;
}

bool asPredictandPrecipitation::BuildDataNormalized() {
    for (int iStat = 0; iStat < m_stationsNb; iStat++) {
        float prt = 1.0;
        if (m_returnPeriodNormalization != 0) {
            prt = GetPrecipitationOfReturnPeriod(iStat, 1, m_returnPeriodNormalization);
        }

        for (int iTime = 0; iTime < m_timeLength; iTime++) {
            if (m_isSqrt) {
                m_dataNormalized(iTime, iStat) = std::sqrt(m_dataRaw(iTime, iStat) / prt);
            } else {
                m_dataNormalized(iTime, iStat) = m_dataRaw(iTime, iStat) / prt;
            }
        }
    }
    return true;
}
