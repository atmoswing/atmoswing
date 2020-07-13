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
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
 */

#include "asResultsForecastAggregator.h"

#include "asFileText.h"
#include "asFileXml.h"

asResultsForecastAggregator::asResultsForecastAggregator() : wxObject() {}

asResultsForecastAggregator::~asResultsForecastAggregator() {
    ClearArrays();
}

bool asResultsForecastAggregator::Add(asResultsForecast *forecast) {
    bool createNewMethodRow = true;

    for (int methodRow = 0; methodRow < (int)m_forecasts.size(); methodRow++) {
        wxASSERT(!m_forecasts[methodRow].empty());
        asResultsForecast *refForecast = m_forecasts[methodRow][0];

        bool compatible = true;

        if (!refForecast->GetMethodId().IsSameAs(forecast->GetMethodId(), false)) compatible = false;
        if (refForecast->GetPredictandParameter() != forecast->GetPredictandParameter()) compatible = false;
        if (refForecast->GetPredictandTemporalResolution() != forecast->GetPredictandTemporalResolution())
            compatible = false;
        if (refForecast->GetPredictandSpatialAggregation() != forecast->GetPredictandSpatialAggregation())
            compatible = false;
        if (!refForecast->GetPredictandDatasetId().IsSameAs(forecast->GetPredictandDatasetId(), false))
            compatible = false;
        if (!refForecast->GetPredictandDatabase().IsSameAs(forecast->GetPredictandDatabase(), false))
            compatible = false;

        if (compatible) {
            // Detailed checks
            if (forecast->IsCompatibleWith(refForecast)) {
                // Check that it is not the exact same forecast
                for (int forecastRow = 0; forecastRow < (int)m_forecasts[methodRow].size(); forecastRow++) {
                    asResultsForecast *otherForecast = m_forecasts[methodRow][forecastRow];
                    if (forecast->IsSameAs(otherForecast)) {
                        wxLogVerbose(_("This forecast has already been loaded."));
                        return false;
                    }
                }

                m_forecasts[methodRow].push_back(forecast);
                m_pastForecasts[methodRow].resize(m_forecasts[methodRow].size());
                createNewMethodRow = false;
                break;
            } else {
                wxLogError(_("The forecast \"%s\" (%s) is not fully compatible with \"%s\" (%s)"),
                           forecast->GetSpecificTagDisplay(), forecast->GetMethodIdDisplay(),
                           refForecast->GetSpecificTagDisplay(), refForecast->GetMethodIdDisplay());
                return false;
            }
        }
    }

    if (createNewMethodRow) {
        m_forecasts.resize(m_forecasts.size() + 1);
        m_pastForecasts.resize(m_pastForecasts.size() + 1);
        m_forecasts[m_forecasts.size() - 1].push_back(forecast);
        m_pastForecasts[m_pastForecasts.size() - 1].resize(1);
    }

    return true;
}

bool asResultsForecastAggregator::AddPastForecast(int methodRow, int forecastRow, asResultsForecast *forecast) {
    bool compatible = true;

    wxASSERT((int)m_forecasts.size() > methodRow);
    wxASSERT((int)m_pastForecasts.size() > methodRow);
    wxASSERT((int)m_forecasts[methodRow].size() > forecastRow);
    wxASSERT((int)m_pastForecasts[methodRow].size() > forecastRow);

    asResultsForecast *refForecast = m_forecasts[methodRow][forecastRow];

    if (!refForecast->GetMethodId().IsSameAs(forecast->GetMethodId(), false)) compatible = false;
    if (!refForecast->GetSpecificTag().IsSameAs(forecast->GetSpecificTag(), false)) compatible = false;
    if (refForecast->GetPredictandParameter() != forecast->GetPredictandParameter()) compatible = false;
    if (refForecast->GetPredictandTemporalResolution() != forecast->GetPredictandTemporalResolution())
        compatible = false;
    if (refForecast->GetPredictandSpatialAggregation() != forecast->GetPredictandSpatialAggregation())
        compatible = false;
    if (!refForecast->GetPredictandDatasetId().IsSameAs(forecast->GetPredictandDatasetId(), false)) compatible = false;
    if (!refForecast->GetPredictandDatabase().IsSameAs(forecast->GetPredictandDatabase(), false)) compatible = false;

    if (compatible) {
        m_pastForecasts[methodRow][forecastRow].push_back(forecast);
    } else {
        wxLogError(_("The past forecast \"%s\" (%s) is not fully compatible with the current version of \"%s\" (%s)"),
                   forecast->GetSpecificTagDisplay(), forecast->GetMethodIdDisplay(),
                   refForecast->GetSpecificTagDisplay(), refForecast->GetMethodIdDisplay());
        return false;
    }

    return true;
}

void asResultsForecastAggregator::ClearArrays() {
    for (int i = 0; i < m_forecasts.size(); i++) {
        for (int j = 0; j < m_forecasts[i].size(); j++) {
            wxDELETE(m_forecasts[i][j]);
        }
    }
    m_forecasts.clear();

    for (int i = 0; i < m_pastForecasts.size(); i++) {
        for (int j = 0; j < m_pastForecasts[i].size(); j++) {
            for (int k = 0; k < m_pastForecasts[i][j].size(); k++) {
                wxDELETE(m_pastForecasts[i][j][k]);
            }
        }
    }
    m_pastForecasts.clear();
}

int asResultsForecastAggregator::GetMethodsNb() const {
    return (int)m_forecasts.size();
}

int asResultsForecastAggregator::GetForecastsNb(int methodRow) const {
    wxASSERT((int)m_forecasts.size() > methodRow);
    return (int)m_forecasts[methodRow].size();
}

int asResultsForecastAggregator::GetPastMethodsNb() const {
    return (int)m_pastForecasts.size();
}

int asResultsForecastAggregator::GetPastForecastsNb(int methodRow) const {
    wxASSERT((int)m_pastForecasts.size() > methodRow);
    return (int)m_pastForecasts[methodRow].size();
}

int asResultsForecastAggregator::GetPastForecastsNb(int methodRow, int forecastRow) const {
    wxASSERT(m_pastForecasts.size() > methodRow);
    wxASSERT(m_pastForecasts[methodRow].size() > forecastRow);
    return (int)m_pastForecasts[methodRow][forecastRow].size();
}

asResultsForecast *asResultsForecastAggregator::GetForecast(int methodRow, int forecastRow) const {
    wxASSERT(m_forecasts.size() > methodRow);
    wxASSERT(m_forecasts[methodRow].size() > forecastRow);
    return m_forecasts[methodRow][forecastRow];
}

asResultsForecast *asResultsForecastAggregator::GetPastForecast(int methodRow, int forecastRow, int leadTimeRow) const {
    wxASSERT(m_pastForecasts.size() > methodRow);
    wxASSERT(m_pastForecasts[methodRow].size() > forecastRow);
    return m_pastForecasts[methodRow][forecastRow][leadTimeRow];
}

wxString asResultsForecastAggregator::GetForecastName(int methodRow, int forecastRow) const {
    wxString name = wxEmptyString;

    if (m_forecasts.empty()) return wxEmptyString;

    wxASSERT(m_forecasts.size() > methodRow);

    if (m_forecasts.size() > methodRow && m_forecasts[methodRow].size() > forecastRow) {
        name = m_forecasts[methodRow][forecastRow]->GetMethodIdDisplay();

        if (!name.IsSameAs(m_forecasts[methodRow][forecastRow]->GetMethodId())) {
            name.Append(wxString::Format(" (%s)", m_forecasts[methodRow][forecastRow]->GetMethodId()));
        }

        if (!m_forecasts[methodRow][forecastRow]->GetSpecificTag().IsEmpty()) {
            name.Append(" - ");
            name.Append(m_forecasts[methodRow][forecastRow]->GetSpecificTagDisplay());
        }
    }

    wxASSERT(!name.IsEmpty());

    return name;
}

wxString asResultsForecastAggregator::GetMethodName(int methodRow) const {
    wxString name = wxEmptyString;

    if (m_forecasts.empty()) return wxEmptyString;

    wxASSERT(m_forecasts.size() > methodRow);

    if (m_forecasts.size() > methodRow && !m_forecasts[methodRow].empty()) {
        name = m_forecasts[methodRow][0]->GetMethodIdDisplay();

        if (!name.IsSameAs(m_forecasts[methodRow][0]->GetMethodId())) {
            name.Append(wxString::Format(" (%s)", m_forecasts[methodRow][0]->GetMethodId()));
        }
    }

    wxASSERT(!name.IsEmpty());

    return name;
}

vwxs asResultsForecastAggregator::GetAllMethodNames() const {
    vwxs names;

    for (int methodRow = 0; methodRow < m_forecasts.size(); methodRow++) {
        wxASSERT(!m_forecasts[methodRow].empty());

        wxString methodName = m_forecasts[methodRow][0]->GetMethodIdDisplay();
        if (!methodName.IsSameAs(m_forecasts[methodRow][0]->GetMethodId())) {
            methodName.Append(wxString::Format(" (%s)", m_forecasts[methodRow][0]->GetMethodId()));
        }
        names.push_back(methodName);
    }

    wxASSERT(!names.empty());

    return names;
}

vwxs asResultsForecastAggregator::GetAllForecastNames() const {
    vwxs names;

    for (int methodRow = 0; methodRow < m_forecasts.size(); methodRow++) {
        wxASSERT(!m_forecasts[methodRow].empty());

        wxString methodName = m_forecasts[methodRow][0]->GetMethodIdDisplay();
        if (!methodName.IsSameAs(m_forecasts[methodRow][0]->GetMethodId())) {
            methodName.Append(wxString::Format(" (%s)", m_forecasts[methodRow][0]->GetMethodId()));
        }

        for (int forecastRow = 0; forecastRow < m_forecasts[methodRow].size(); forecastRow++) {
            wxString name = methodName;

            if (!m_forecasts[methodRow][forecastRow]->GetSpecificTag().IsEmpty()) {
                name.Append(" - ");
                name.Append(m_forecasts[methodRow][forecastRow]->GetSpecificTagDisplay());
            }
            names.push_back(name);
        }
    }

    wxASSERT(!names.empty());

    return names;
}

wxArrayString asResultsForecastAggregator::GetAllForecastNamesWxArray() const {
    wxArrayString names;

    for (int methodRow = 0; methodRow < m_forecasts.size(); methodRow++) {
        wxASSERT(!m_forecasts[methodRow].empty());

        wxString methodName = m_forecasts[methodRow][0]->GetMethodIdDisplay();
        if (!methodName.IsSameAs(m_forecasts[methodRow][0]->GetMethodId())) {
            methodName.Append(wxString::Format(" (%s)", m_forecasts[methodRow][0]->GetMethodId()));
        }

        for (int forecastRow = 0; forecastRow < m_forecasts[methodRow].size(); forecastRow++) {
            wxString name = methodName;

            if (!m_forecasts[methodRow][forecastRow]->GetSpecificTag().IsEmpty()) {
                name.Append(" - ");
                name.Append(m_forecasts[methodRow][forecastRow]->GetSpecificTagDisplay());
            }
            names.Add(name);
        }
    }

    wxASSERT(!names.empty());

    return names;
}

wxString asResultsForecastAggregator::GetFilePath(int methodRow, int forecastRow) const {
    if (forecastRow < 0) {
        forecastRow = 0;
    }

    return m_forecasts[methodRow][forecastRow]->GetFilePath();
}

a1f asResultsForecastAggregator::GetTargetDates(int methodRow) const {
    double firstDate = 9999999999, lastDate = 0;

    wxASSERT(methodRow >= 0);

    for (int forecastRow = 0; forecastRow < m_forecasts[methodRow].size(); forecastRow++) {
        a1f fcastDates = m_forecasts[methodRow][forecastRow]->GetTargetDates();
        if (fcastDates[0] < firstDate) {
            firstDate = fcastDates[0];
        }
        if (fcastDates[fcastDates.size() - 1] > lastDate) {
            lastDate = fcastDates[fcastDates.size() - 1];
        }
    }

    int size = asRound(lastDate - firstDate + 1);
    a1f dates = a1f::LinSpaced(size, firstDate, lastDate);

    return dates;
}

a1f asResultsForecastAggregator::GetTargetDates(int methodRow, int forecastRow) const {
    return m_forecasts[methodRow][forecastRow]->GetTargetDates();
}

a1f asResultsForecastAggregator::GetFullTargetDates() const {
    double firstDate = 9999999999, lastDate = 0;

    for (int methodRow = 0; methodRow < m_forecasts.size(); methodRow++) {
        for (int forecastRow = 0; forecastRow < m_forecasts[methodRow].size(); forecastRow++) {
            a1f fcastDates = m_forecasts[methodRow][forecastRow]->GetTargetDates();
            if (fcastDates[0] < firstDate) {
                firstDate = fcastDates[0];
            }
            if (fcastDates[fcastDates.size() - 1] > lastDate) {
                lastDate = fcastDates[fcastDates.size() - 1];
            }
        }
    }

    int size = asRound(lastDate - firstDate + 1);
    a1f dates = a1f::LinSpaced(size, firstDate, lastDate);

    return dates;
}

int asResultsForecastAggregator::GetForecastRowSpecificForStationId(int methodRow, int stationId) const {
    if (GetForecastsNb(methodRow) == 1) return 0;

    // Pick up the most relevant forecast for the station
    for (int i = 0; i < GetForecastsNb(methodRow); i++) {
        asResultsForecast *forecast = m_forecasts[methodRow][i];
        if (forecast->IsSpecificForStationId(stationId)) {
            return i;
        }
    }

    wxLogWarning(_("No specific forecast was found for station ID %d"), stationId);

    return 0;
}

int asResultsForecastAggregator::GetForecastRowSpecificForStationRow(int methodRow, int stationRow) const {
    if (GetForecastsNb(methodRow) == 1) return 0;

    // Pick up the most relevant forecast for the station
    for (int i = 0; i < GetForecastsNb(methodRow); i++) {
        asResultsForecast *forecast = m_forecasts[methodRow][i];
        int stationId = forecast->GetStationId(stationRow);
        if (forecast->IsSpecificForStationId(stationId)) {
            return i;
        }
    }

    wxLogWarning(_("No specific forecast was found for station n°%d"), stationRow);

    return 0;
}

wxArrayString asResultsForecastAggregator::GetStationNames(int methodRow, int forecastRow) const {
    wxArrayString stationNames;

    if (m_forecasts.empty()) return stationNames;

    wxASSERT(m_forecasts.size() > methodRow);
    wxASSERT(m_forecasts[methodRow].size() > forecastRow);

    stationNames = m_forecasts[methodRow][forecastRow]->GetStationNamesWxArrayString();

    return stationNames;
}

wxString asResultsForecastAggregator::GetStationName(int methodRow, int forecastRow, int stationRow) const {
    wxString stationName;

    if (m_forecasts.empty()) return wxEmptyString;

    wxASSERT(m_forecasts.size() > methodRow);
    wxASSERT(m_forecasts[methodRow].size() > forecastRow);

    stationName = m_forecasts[methodRow][forecastRow]->GetStationName(stationRow);

    return stationName;
}

wxArrayString asResultsForecastAggregator::GetStationNamesWithHeights(int methodRow, int forecastRow) const {
    wxArrayString stationNames;

    if (m_forecasts.empty()) return stationNames;

    wxASSERT(m_forecasts.size() > methodRow);
    wxASSERT(m_forecasts[methodRow].size() > forecastRow);

    stationNames = m_forecasts[methodRow][forecastRow]->GetStationNamesAndHeightsWxArrayString();

    return stationNames;
}

wxString asResultsForecastAggregator::GetStationNameWithHeight(int methodRow, int forecastRow, int stationRow) const {
    wxString stationName;

    if (m_forecasts.empty()) return wxEmptyString;

    wxASSERT(m_forecasts.size() > methodRow);
    wxASSERT(m_forecasts[methodRow].size() > forecastRow);

    stationName = m_forecasts[methodRow][forecastRow]->GetStationNameAndHeight(stationRow);

    return stationName;
}

int asResultsForecastAggregator::GetLeadTimeLength(int methodRow, int forecastRow) const {
    if (m_forecasts.empty()) return 0;

    wxASSERT(m_forecasts.size() > methodRow);
    wxASSERT(m_forecasts[methodRow].size() > forecastRow);

    int length = m_forecasts[methodRow][forecastRow]->GetTargetDatesLength();

    wxASSERT(length > 0);

    return length;
}

int asResultsForecastAggregator::GetLeadTimeLengthMax() const {
    if (m_forecasts.empty()) return 0;

    int length = 0;

    for (int i = 0; i < (int)m_forecasts.size(); i++) {
        for (int j = 0; j < (int)m_forecasts[i].size(); j++) {
            length = wxMax(length, m_forecasts[i][j]->GetTargetDatesLength());
        }
    }

    return length;
}

wxArrayString asResultsForecastAggregator::GetLeadTimes(int methodRow, int forecastRow) const {
    wxArrayString leadTimes;

    if (m_forecasts.empty()) return leadTimes;

    wxASSERT(m_forecasts.size() > methodRow);
    wxASSERT(m_forecasts[methodRow].size() > forecastRow);

    a1f dates = m_forecasts[methodRow][forecastRow]->GetTargetDates();

    for (int i = 0; i < dates.size(); i++) {
        leadTimes.Add(asTime::GetStringTime(dates[i], "DD.MM.YYYY HH"));
    }

    return leadTimes;
}

a1f asResultsForecastAggregator::GetMethodMaxValues(a1f &dates, int methodRow, int returnPeriodRef,
                                                    float quantileThreshold) const {
    wxASSERT(returnPeriodRef >= 2);
    wxASSERT(quantileThreshold > 0);
    wxASSERT(quantileThreshold < 1);
    if (returnPeriodRef < 2) returnPeriodRef = 2;
    if (quantileThreshold <= 0) quantileThreshold = (float)0.9;
    if (quantileThreshold > 1) quantileThreshold = (float)0.9;

    wxASSERT((int)m_forecasts.size() > methodRow);

    a1f maxValues = a1f::Ones(dates.size());
    maxValues *= NaNf;

    bool singleMethod = (GetForecastsNb(methodRow) == 1);

    for (int forecastRow = 0; forecastRow < (int)m_forecasts[methodRow].size(); forecastRow++) {
        asResultsForecast *forecast = m_forecasts[methodRow][forecastRow];

        // Get return period index
        int indexReferenceAxis = asNOT_FOUND;
        if (forecast->HasReferenceValues()) {
            a1f forecastReferenceAxis = forecast->GetReferenceAxis();
            indexReferenceAxis = asFind(&forecastReferenceAxis[0],
                                        &forecastReferenceAxis[forecastReferenceAxis.size() - 1], returnPeriodRef);
            if ((indexReferenceAxis == asNOT_FOUND) || (indexReferenceAxis == asOUT_OF_RANGE)) {
                wxLogError(_("The desired return period is not available in the forecast file."));
            }
        }

        // Check lead times effectively available for the current forecast
        int leadtimeMin = 0;
        int leadtimeMax = dates.size() - 1;

        a1f availableDates = forecast->GetTargetDates();

        while (dates[leadtimeMin] < availableDates[0]) {
            leadtimeMin++;
        }
        while (dates[leadtimeMax] > availableDates[availableDates.size() - 1]) {
            leadtimeMax--;
        }
        wxASSERT(leadtimeMin < leadtimeMax);

        // Get the values of the relevant stations only
        vi relevantStations;
        if (singleMethod) {
            a1i relevantStationsTmp = forecast->GetStationIds();
            for (int i = 0; i < relevantStationsTmp.size(); i++) {
                relevantStations.push_back(relevantStationsTmp[i]);
            }
        } else {
            relevantStations = forecast->GetPredictandStationIds();
        }

        for (int iStat = 0; iStat < (int)relevantStations.size(); iStat++) {
            int indexStation = forecast->GetStationRowFromId(relevantStations[iStat]);

            // Get values for return period
            float factor = 1;
            if (forecast->HasReferenceValues()) {
                float precip = forecast->GetReferenceValue(indexStation, indexReferenceAxis);
                wxASSERT(precip > 0);
                wxASSERT(precip < 500);
                factor = 1.0 / precip;
                wxASSERT(factor > 0);
            }

            for (int iLead = leadtimeMin; iLead <= leadtimeMax; iLead++) {
                if (asIsNaN(maxValues[iLead])) {
                    maxValues[iLead] = -999999;
                }

                float thisVal = 0;

                // Get values
                a1f theseVals = forecast->GetAnalogsValuesRaw(iLead, indexStation);

                // Process quantiles
                if (asHasNaN(&theseVals[0], &theseVals[theseVals.size() - 1])) {
                    thisVal = NaNf;
                } else {
                    float forecastVal = asGetValueForQuantile(theseVals, quantileThreshold);
                    forecastVal *= factor;
                    thisVal = forecastVal;
                }

                // Keep it if higher
                if (thisVal > maxValues[iLead]) {
                    maxValues[iLead] = thisVal;
                }
            }
        }
    }

    return maxValues;
}

a1f asResultsForecastAggregator::GetOverallMaxValues(a1f &dates, int returnPeriodRef, float quantileThreshold) const {
    a2f allMax(dates.size(), m_forecasts.size());

    for (int methodRow = 0; methodRow < (int)m_forecasts.size(); methodRow++) {
        allMax.col(methodRow) = GetMethodMaxValues(dates, methodRow, returnPeriodRef, quantileThreshold);
    }

    // Extract the highest values
    a1f values = allMax.rowwise().maxCoeff();

    return values;
}

bool asResultsForecastAggregator::ExportSyntheticFullXml(const wxString &dirPath) const {
    // Quantile values
    a1f quantiles(3);
    quantiles << 20, 60, 90;

    // Create 1 file per method
    for (int methodRow = 0; methodRow < (int)m_forecasts.size(); methodRow++) {
        // Filename
        wxString filePath = dirPath;
        filePath.Append(DS);
        wxString dirstructure = "YYYY";
        dirstructure.Append(DS).Append("MM").Append(DS).Append("DD");
        wxString directory = asTime::GetStringTime(m_forecasts[methodRow][0]->GetLeadTimeOrigin(), dirstructure);
        filePath.Append(directory).Append(DS);
        wxString forecastname = m_forecasts[methodRow][0]->GetMethodId();
        wxString nowstr = asTime::GetStringTime(m_forecasts[methodRow][0]->GetLeadTimeOrigin(), "YYYY_MM_DD_hh");
        wxString ext = "xml";
        wxString filename = wxString::Format("%s.%s.%s", nowstr, forecastname, ext);
        filePath.Append(filename);

        // Create file
        asFileXml fileExport(filePath, asFile::Replace);
        if (!fileExport.Open()) return false;

        // General attributes
        fileExport.GetRoot()->AddAttribute("created", asTime::GetStringTime(asTime::NowMJD(), "DD.MM.YYYY HH"));

        // Method description
        wxXmlNode *nodeMethod = new wxXmlNode(wxXML_ELEMENT_NODE, "method");
        nodeMethod->AddChild(fileExport.CreateNodeWithValue("id", m_forecasts[methodRow][0]->GetMethodId()));
        nodeMethod->AddChild(fileExport.CreateNodeWithValue("name", m_forecasts[methodRow][0]->GetMethodIdDisplay()));
        nodeMethod->AddChild(fileExport.CreateNodeWithValue("description", m_forecasts[methodRow][0]->GetDescription()));
        fileExport.AddChild(nodeMethod);

        // Reference axis
        if (m_forecasts[methodRow][0]->HasReferenceValues()) {
            a1f refAxis = m_forecasts[methodRow][0]->GetReferenceAxis();
            wxXmlNode *nodeReferenceAxis = new wxXmlNode(wxXML_ELEMENT_NODE, "reference_axis");
            for (int i = 0; i < refAxis.size(); i++) {
                nodeReferenceAxis->AddChild(fileExport.CreateNodeWithValue("reference",
                                            wxString::Format("%.2f", refAxis[i])));
            }
            fileExport.AddChild(nodeReferenceAxis);
        }

        // Target dates
        a1f targetDates = m_forecasts[methodRow][0]->GetTargetDates();
        wxXmlNode *nodeTargetDates = new wxXmlNode(wxXML_ELEMENT_NODE, "target_dates");
        for (int i = 0; i < targetDates.size(); i++) {
            wxXmlNode *nodeTargetDate = new wxXmlNode(wxXML_ELEMENT_NODE, "target_date");
            nodeTargetDate->AddChild(fileExport.CreateNodeWithValue("date", asTime::GetStringTime(targetDates[i], "DD.MM.YYYY HH")));
            nodeTargetDate->AddChild(fileExport.CreateNodeWithValue("analogs_nb", m_forecasts[methodRow][0]->GetAnalogsNumber(i)));
            nodeTargetDates->AddChild(nodeTargetDate);
        }
        fileExport.AddChild(nodeTargetDates);

        // Quantiles
        wxXmlNode *nodeQuantiles = new wxXmlNode(wxXML_ELEMENT_NODE, "quantile_names");
        for (int i = 0; i < quantiles.size(); i++) {
            nodeQuantiles->AddChild(fileExport.CreateNodeWithValue("quantile", wxString::Format("%d", (int)quantiles[i])));
        }
        fileExport.AddChild(nodeQuantiles);

        // Results per station
        a1i stationIds = m_forecasts[methodRow][0]->GetStationIds();
        a2f referenceValues;
        if (m_forecasts[methodRow][0]->HasReferenceValues()) {
            referenceValues = m_forecasts[methodRow][0]->GetReferenceValues();
            wxASSERT(referenceValues.rows() == stationIds.size());
        }
        wxXmlNode *nodeStations = new wxXmlNode(wxXML_ELEMENT_NODE, "stations");
        for (int i = 0; i < stationIds.size(); i++) {
            // Get specific forecast
            int forecastRow = GetForecastRowSpecificForStationId(methodRow, stationIds[i]);
            asResultsForecast *forecast = m_forecasts[methodRow][forecastRow];

            // Set station properties
            wxXmlNode *nodeStation = new wxXmlNode(wxXML_ELEMENT_NODE, "station");
            nodeStation->AddChild(fileExport.CreateNodeWithValue("id", stationIds[i]));
            nodeStation->AddChild(fileExport.CreateNodeWithValue("official_id", forecast->GetStationOfficialId(i)));
            nodeStation->AddChild(fileExport.CreateNodeWithValue("name", forecast->GetStationName(i)));
            nodeStation->AddChild(fileExport.CreateNodeWithValue("x", forecast->GetStationXCoord(i)));
            nodeStation->AddChild(fileExport.CreateNodeWithValue("y", forecast->GetStationYCoord(i)));
            nodeStation->AddChild(fileExport.CreateNodeWithValue("height", forecast->GetStationHeight(i)));
            nodeStation->AddChild(fileExport.CreateNodeWithValue("specific_parameters", forecast->GetSpecificTagDisplay()));

            // Set reference values
            if (forecast->HasReferenceValues()) {
                wxXmlNode *nodeReferenceValues = new wxXmlNode(wxXML_ELEMENT_NODE, "reference_values");
                for (int j = 0; j < referenceValues.cols(); j++) {
                    nodeReferenceValues->AddChild(
                        fileExport.CreateNodeWithValue("value", wxString::Format("%.2f", referenceValues(i, j))));
                }
                nodeStation->AddChild(nodeReferenceValues);
            }

            // Set 10 best analogs
            wxXmlNode *nodeBestAnalogs = new wxXmlNode(wxXML_ELEMENT_NODE, "best_analogs");
            for (int j = 0; j < targetDates.size(); j++) {
                a1f analogValues = forecast->GetAnalogsValuesRaw(j, i);
                a1f analogDates = forecast->GetAnalogsDates(j);
                a1f analogCriteria = forecast->GetAnalogsCriteria(j);
                wxASSERT(analogValues.size() == analogDates.size());
                wxASSERT(analogValues.size() == analogCriteria.size());

                wxXmlNode *nodeTargetDate = new wxXmlNode(wxXML_ELEMENT_NODE, "target_date");
                for (int k = 0; k < wxMin(10, analogValues.size()); k++) {
                    wxXmlNode *nodeAnalog = new wxXmlNode(wxXML_ELEMENT_NODE, "analog");
                    nodeAnalog->AddChild(fileExport.CreateNodeWithValue("date", asTime::GetStringTime(analogDates[k], "DD.MM.YYYY HH")));
                    nodeAnalog->AddChild(fileExport.CreateNodeWithValue("value", wxString::Format("%.1f", analogValues[k])));
                    nodeAnalog->AddChild(fileExport.CreateNodeWithValue("criteria", wxString::Format("%.1f", analogCriteria[k])));

                    nodeTargetDate->AddChild(nodeAnalog);
                }
                nodeBestAnalogs->AddChild(nodeTargetDate);
            }
            nodeStation->AddChild(nodeBestAnalogs);

            // Set quantiles
            wxXmlNode *nodeAnalogsQuantiles = new wxXmlNode(wxXML_ELEMENT_NODE, "analogs_quantiles");
            for (int j = 0; j < targetDates.size(); j++) {
                a1f analogValues = forecast->GetAnalogsValuesRaw(j, i);

                wxXmlNode *nodeTargetDate = new wxXmlNode(wxXML_ELEMENT_NODE, "target_date");
                for (int k = 0; k < wxMin(10, quantiles.size()); k++) {
                    float pcVal = asGetValueForQuantile(analogValues, quantiles[k] / 100);
                    nodeTargetDate->AddChild(fileExport.CreateNodeWithValue("quantile", wxString::Format("%.1f", pcVal)));
                }
                nodeAnalogsQuantiles->AddChild(nodeTargetDate);
            }
            nodeStation->AddChild(nodeAnalogsQuantiles);

            // Set mean
            wxXmlNode *nodeAnalogsMean = new wxXmlNode(wxXML_ELEMENT_NODE, "analogs_mean");
            for (int j = 0; j < targetDates.size(); j++) {
                a1f analogValues = forecast->GetAnalogsValuesRaw(j, i);
                wxXmlNode *nodeTargetDate = new wxXmlNode(wxXML_ELEMENT_NODE, "target_date");
                float mean = analogValues.mean();
                nodeTargetDate->AddChild(fileExport.CreateNodeWithValue("mean", wxString::Format("%.1f", mean)));
                nodeAnalogsMean->AddChild(nodeTargetDate);
            }
            nodeStation->AddChild(nodeAnalogsMean);

            nodeStations->AddChild(nodeStation);
        }
        fileExport.AddChild(nodeStations);

        fileExport.Save();
    }

    return true;
}

bool asResultsForecastAggregator::ExportSyntheticSmallCsv(const wxString &dirPath) const {
    // Quantile values
    a1f quantiles(6);
    quantiles << 0, 25, 50, 75, 90, 100;

    // Create 1 file per method
    for (int methodRow = 0; methodRow < (int)m_forecasts.size(); methodRow++) {
        // Filename
        wxString filePath = dirPath;
        filePath.Append(DS);
        wxString dirstructure = "YYYY";
        dirstructure.Append(DS).Append("MM").Append(DS).Append("DD");
        wxString directory = asTime::GetStringTime(m_forecasts[methodRow][0]->GetLeadTimeOrigin(), dirstructure);
        filePath.Append(directory).Append(DS);
        wxString forecastname = m_forecasts[methodRow][0]->GetMethodId();
        wxString nowstr = asTime::GetStringTime(m_forecasts[methodRow][0]->GetLeadTimeOrigin(), "YYYY_MM_DD_hh");
        wxString filename = wxString::Format("%s.%s.%s", nowstr, forecastname, "txt");
        filePath.Append(filename);

        // Create file
        asFileText fileExport(filePath, asFile::Replace);
        if (!fileExport.Open()) return false;

        // Method description
        fileExport.AddContent(wxString::Format("method: %s | %s\n",
                                               m_forecasts[methodRow][0]->GetMethodId(),
                                               m_forecasts[methodRow][0]->GetMethodIdDisplay()));

        // Results per station
        a1i stationIds = m_forecasts[methodRow][0]->GetStationIds();

        for (int i = 0; i < stationIds.size(); i++) {
            // Get specific forecast
            int forecastRow = GetForecastRowSpecificForStationId(methodRow, stationIds[i]);
            asResultsForecast *forecast = m_forecasts[methodRow][forecastRow];

            // Set station properties
            fileExport.AddContent(wxString::Format("station %d: %s (%s)\n",
                                                   stationIds[i],
                                                   forecast->GetStationName(i),
                                                   forecast->GetSpecificTagDisplay()));
        }
        fileExport.AddContent("\n");

        // Quantiles
        wxString headerQuantiles;
        for (int k = 0; k < quantiles.size(); k++) {
            headerQuantiles.Append(wxString::Format("q%d raw; q%d transformed; ", (int)quantiles[k], (int)quantiles[k]));
        }

        // Headers
        fileExport.AddContent(wxString::Format("station; lead time; nb analogs; mean raw; mean transformed; %s\n", headerQuantiles));

        for (int i = 0; i < stationIds.size(); i++) {
            // Get specific forecast
            int forecastRow = GetForecastRowSpecificForStationId(methodRow, stationIds[i]);
            asResultsForecast *forecast = m_forecasts[methodRow][forecastRow];

            // Target dates
            a1f targetDates = forecast->GetTargetDates();
            for (int j = 0; j < targetDates.size(); j++) {
                a1f analogValuesRaw = forecast->GetAnalogsValuesRaw(j, i);
                a1f analogValuesNorm = forecast->GetAnalogsValuesNorm(j, i);

                wxString valuesQuantiles;
                for (int k = 0; k < quantiles.size(); k++) {
                    float pcValRaw = asGetValueForQuantile(analogValuesRaw, quantiles[k] / 100);
                    float pcValNorm = asGetValueForQuantile(analogValuesNorm, quantiles[k] / 100);
                    valuesQuantiles.Append(wxString::Format("%.2f; %.2f; ", pcValRaw, pcValNorm));
                }

                fileExport.AddContent(wxString::Format("%d; %s; %d; %.2f; %.2f; %s\n",
                                                       stationIds[i],
                                                       asTime::GetStringTime(targetDates[j], "DD.MM.YYYY HH"),
                                                       forecast->GetAnalogsNumber(j),
                                                       analogValuesRaw.mean(),
                                                       analogValuesNorm.mean(),
                                                       valuesQuantiles));
            }
        }

        fileExport.Close();
    }

    return true;
}

bool asResultsForecastAggregator::ExportSyntheticCustomCsvFVG(const wxString &dirPath) const {
    // Quantile values
    a1f quantiles(6);
    quantiles << 0, 25, 50, 75, 90, 100;

    // Create 1 file per method
    for (int methodRow = 0; methodRow < (int)m_forecasts.size(); methodRow++) {
        // Filename
        wxString filePath = dirPath;
        filePath.Append(DS);
        wxString dirstructure = "YYYY";
        dirstructure.Append(DS).Append("MM").Append(DS).Append("DD");
        wxString directory = asTime::GetStringTime(m_forecasts[methodRow][0]->GetLeadTimeOrigin(), dirstructure);
        filePath.Append(directory).Append(DS);
        wxString forecastname = m_forecasts[methodRow][0]->GetMethodId();
        wxString nowstr = asTime::GetStringTime(m_forecasts[methodRow][0]->GetLeadTimeOrigin(), "YYYY_MM_DD_hh");
        wxString filename = wxString::Format("%s.%s.%s", nowstr, forecastname, "txt");
        filePath.Append(filename);

        // Create file
        asFileText fileExport(filePath, asFile::Replace);
        if (!fileExport.Open()) return false;

        // Results per station
        a1i stationIds = m_forecasts[methodRow][0]->GetStationIds();

        // Quantiles
        wxString headerQuantiles;
        for (int k = 0; k < quantiles.size(); k++) {
            headerQuantiles.Append(wxString::Format("q%d; ", (int)quantiles[k]));
        }

        // Headers
        fileExport.AddContent(wxString::Format("area; lead time; forecast; nb analogs; mean raw; mean transformed; %s\n", headerQuantiles));

        for (int i = 0; i < stationIds.size(); i++) {
            // Get specific forecast
            int forecastRow = GetForecastRowSpecificForStationId(methodRow, stationIds[i]);
            asResultsForecast *forecast = m_forecasts[methodRow][forecastRow];

            // Target dates
            a1f targetDates = forecast->GetTargetDates();
            for (int j = 0; j < targetDates.size(); j++) {
                a1f analogValuesRaw = forecast->GetAnalogsValuesRaw(j, i);
                a1f analogValuesNorm = forecast->GetAnalogsValuesNorm(j, i);

                wxString valuesQuantiles;
                for (int k = 0; k < quantiles.size(); k++) {
                    float pcValNorm = asGetValueForQuantile(analogValuesNorm, quantiles[k] / 100);
                    valuesQuantiles.Append(wxString::Format("%.2f; ", pcValNorm));
                }

                int hourForecast = int (24 * (targetDates[j] - forecast->GetLeadTimeOrigin()));

                fileExport.AddContent(wxString::Format("%d; %s; %02d; %d; %.2f; %.2f; %s\n",
                                                       stationIds[i] - 2,
                                                       asTime::GetStringTime(targetDates[j], "YYYYMMDDHH"),
                                                       hourForecast,
                                                       forecast->GetAnalogsNumber(j),
                                                       analogValuesRaw.mean(),
                                                       analogValuesNorm.mean(),
                                                       valuesQuantiles));
            }
        }

        fileExport.Close();
    }

    return true;
}