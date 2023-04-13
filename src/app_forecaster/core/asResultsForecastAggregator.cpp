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

asResultsForecastAggregator::asResultsForecastAggregator()
    : wxObject() {}

asResultsForecastAggregator::~asResultsForecastAggregator() {
    ClearArrays();
}

bool asResultsForecastAggregator::Add(asResultsForecast* forecast) {
    bool createNewMethodRow = true;

    for (int methodRow = 0; methodRow < (int)m_forecasts.size(); methodRow++) {
        wxASSERT(!m_forecasts[methodRow].empty());
        asResultsForecast* refForecast = m_forecasts[methodRow][0];

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
                    asResultsForecast* otherForecast = m_forecasts[methodRow][forecastRow];
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

bool asResultsForecastAggregator::AddPastForecast(int methodRow, int forecastRow, asResultsForecast* forecast) {
    bool compatible = true;

    wxASSERT((int)m_forecasts.size() > methodRow);
    wxASSERT((int)m_pastForecasts.size() > methodRow);
    wxASSERT((int)m_forecasts[methodRow].size() > forecastRow);
    wxASSERT((int)m_pastForecasts[methodRow].size() > forecastRow);

    asResultsForecast* refForecast = m_forecasts[methodRow][forecastRow];

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

asResultsForecast* asResultsForecastAggregator::GetForecast(int methodRow, int forecastRow) const {
    wxASSERT(m_forecasts.size() > methodRow);
    wxASSERT(m_forecasts[methodRow].size() > forecastRow);
    return m_forecasts[methodRow][forecastRow];
}

asResultsForecast* asResultsForecastAggregator::GetPastForecast(int methodRow, int forecastRow, int leadTimeRow) const {
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
            name.Append(asStrF(" (%s)", m_forecasts[methodRow][forecastRow]->GetMethodId()));
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
            name.Append(asStrF(" (%s)", m_forecasts[methodRow][0]->GetMethodId()));
        }
    }

    wxASSERT(!name.IsEmpty());

    return name;
}

vwxs asResultsForecastAggregator::GetMethodNames() const {
    vwxs names;

    for (int methodRow = 0; methodRow < m_forecasts.size(); methodRow++) {
        wxASSERT(!m_forecasts[methodRow].empty());

        wxString methodName = m_forecasts[methodRow][0]->GetMethodIdDisplay();
        if (!methodName.IsSameAs(m_forecasts[methodRow][0]->GetMethodId())) {
            methodName.Append(asStrF(" (%s)", m_forecasts[methodRow][0]->GetMethodId()));
        }
        names.push_back(methodName);
    }

    wxASSERT(!names.empty());

    return names;
}

wxArrayString asResultsForecastAggregator::GetMethodNamesWxArray() const {
    wxArrayString names;

    for (int methodRow = 0; methodRow < m_forecasts.size(); methodRow++) {
        wxASSERT(!m_forecasts[methodRow].empty());

        wxString methodName = m_forecasts[methodRow][0]->GetMethodIdDisplay();
        if (!methodName.IsSameAs(m_forecasts[methodRow][0]->GetMethodId())) {
            methodName.Append(asStrF(" (%s)", m_forecasts[methodRow][0]->GetMethodId()));
        }
        names.Add(methodName);
    }

    wxASSERT(!names.empty());

    return names;
}

wxArrayString asResultsForecastAggregator::GetForecastNamesWxArray(int methodRow) const {
    wxASSERT(m_forecasts.size() > methodRow);
    wxArrayString names;

    for (auto forecast : m_forecasts[methodRow]) {
        if (!forecast->GetSpecificTagDisplay().IsEmpty()) {
            names.Add(forecast->GetSpecificTagDisplay());
        } else {
            names.Add(forecast->GetSpecificTag());
        }
    }

    wxASSERT(!names.empty());

    return names;
}

wxArrayString asResultsForecastAggregator::GetCombinedForecastNamesWxArray() const {
    wxArrayString names;

    for (const auto & method : m_forecasts) {
        wxASSERT(!method.empty());

        wxString methodName = method[0]->GetMethodIdDisplay();
        if (!methodName.IsSameAs(method[0]->GetMethodId())) {
            methodName.Append(asStrF(" (%s)", method[0]->GetMethodId()));
        }

        for (auto forecast : method) {
            wxString name = methodName;

            if (!forecast->GetSpecificTag().IsEmpty()) {
                name.Append(" - ");
                name.Append(forecast->GetSpecificTagDisplay());
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
    float firstDate = 1.0E+9, lastDate = 0;

    for (const auto& forecastGroup : m_forecasts) {
        for (auto forecast : forecastGroup) {
            a1f fcastDates = forecast->GetTargetDates();
            firstDate = wxMin(fcastDates[0], firstDate);
            lastDate = wxMax(fcastDates[fcastDates.size() - 1], lastDate);
        }
    }

    int size = int(asRound(lastDate - firstDate + 1));
    a1f dates = a1f::LinSpaced(size, firstDate, lastDate);

    return dates;
}

int asResultsForecastAggregator::GetForecastRowSpecificForStationId(int methodRow, int stationId) const {
    if (GetForecastsNb(methodRow) == 1) return 0;

    // Pick up the most relevant forecast for the station
    for (int i = 0; i < GetForecastsNb(methodRow); i++) {
        asResultsForecast* forecast = m_forecasts[methodRow][i];
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
        asResultsForecast* forecast = m_forecasts[methodRow][i];
        int stationId = forecast->GetStationId(stationRow);
        if (forecast->IsSpecificForStationId(stationId)) {
            return i;
        }
    }

    wxLogWarning(_("No specific forecast was found for station %d"), stationRow);

    return 0;
}

wxArrayString asResultsForecastAggregator::GetStationNames(int methodRow, int forecastRow) const {
    wxArrayString stationNames;

    if (m_forecasts.empty()) return stationNames;

    wxASSERT(m_forecasts.size() > methodRow);
    wxASSERT(m_forecasts[methodRow].size() > forecastRow);

    stationNames = m_forecasts[methodRow][forecastRow]->GetStationNamesWxArray();

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

    stationNames = m_forecasts[methodRow][forecastRow]->GetStationNamesAndHeightsWxArray();

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

wxArrayString asResultsForecastAggregator::GetTargetDatesWxArray(int methodRow, int forecastRow) const {
    wxArrayString leadTimes;

    if (m_forecasts.empty()) return leadTimes;

    wxASSERT(m_forecasts.size() > methodRow);
    wxASSERT(m_forecasts[methodRow].size() > forecastRow);

    return m_forecasts[methodRow][forecastRow]->GetTargetDatesWxArray();;
}

vf asResultsForecastAggregator::GetMaxExtent() const {
    if (m_forecasts.empty() || m_forecasts[0].empty()) {
        return {0, 0, 0, 0};
    }

    wxASSERT(m_forecasts[0][0]);
    vf vecLonMin = m_forecasts[0][0]->GetPredictorLonMin();
    vf vecLonMax = m_forecasts[0][0]->GetPredictorLonMax();
    vf vecLatMin = m_forecasts[0][0]->GetPredictorLatMin();
    vf vecLatMax = m_forecasts[0][0]->GetPredictorLatMax();

    if (vecLonMin.empty() || vecLonMax.empty() || vecLatMin.empty() || vecLatMax.empty()) {
        return {0, 0, 0, 0};
    }

    vf extent = {
        *std::min_element(vecLonMin.begin(), vecLonMin.end()),
        *std::max_element(vecLonMax.begin(), vecLonMax.end()),
        *std::min_element(vecLatMin.begin(), vecLatMin.end()),
        *std::max_element(vecLatMax.begin(), vecLatMax.end())};

    for (const auto& method : m_forecasts) {
        for (const auto& forecast : method) {
            vecLonMin = forecast->GetPredictorLonMin();
            vecLonMax = forecast->GetPredictorLonMax();
            vecLatMin = forecast->GetPredictorLatMin();
            vecLatMax = forecast->GetPredictorLatMax();
            extent[0] = wxMin(extent[0], *std::min_element(vecLonMin.begin(), vecLonMin.end()));
            extent[1] = wxMax(extent[1], *std::max_element(vecLonMax.begin(), vecLonMax.end()));
            extent[2] = wxMin(extent[2], *std::min_element(vecLatMin.begin(), vecLatMin.end()));
            extent[3] = wxMax(extent[3], *std::max_element(vecLatMax.begin(), vecLatMax.end()));
        }
    }

    return extent;
}

a1f asResultsForecastAggregator::GetMethodMaxValues(a1f& dates, int methodRow, int returnPeriodRef,
                                                    float quantileThreshold) const {
    wxASSERT(returnPeriodRef >= 2);
    wxASSERT(quantileThreshold > 0);
    wxASSERT(quantileThreshold < 1);
    if (returnPeriodRef < 2) returnPeriodRef = 2;
    if (quantileThreshold <= 0) quantileThreshold = (float)0.9;
    if (quantileThreshold > 1) quantileThreshold = (float)0.9;

    wxASSERT((int)m_forecasts.size() > methodRow);

    double timeStep = 1;
    for (auto forecast : m_forecasts[methodRow]) {
        timeStep = wxMin(timeStep, forecast->GetForecastTimeStepHours() / 24.0);
    }

    a1f datesForecast = dates;
    bool timeShiftEndAccumulation = false;
    if (timeStep < 1) {
        timeShiftEndAccumulation = true;
        int datesNb = dates.size() / timeStep;
        datesForecast = a1f::LinSpaced(datesNb, dates[0], dates[0] + (datesNb - 1) * timeStep);
    }

    a1f maxValues = a1f::Ones(datesForecast.size());
    maxValues *= NAN;

    bool singleMethod = (GetForecastsNb(methodRow) == 1);

    for (auto forecast : m_forecasts[methodRow]) {
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
        int leadtimeMax = datesForecast.size() - 1;

        a1f availableDates = forecast->GetTargetDates();

        while (datesForecast[leadtimeMin] < availableDates[0]) {
            leadtimeMin++;
        }
        while (datesForecast[leadtimeMax] > availableDates[availableDates.size() - 1]) {
            leadtimeMax--;
        }
        wxASSERT(leadtimeMin < leadtimeMax);

        // Get the values of the relevant stations only
        vi relevantStations;
        if (singleMethod) {
            a1i relevantStationsTmp = forecast->GetStationIds();
            for (int station : relevantStationsTmp) {
                relevantStations.push_back(station);
            }
        } else {
            relevantStations = forecast->GetPredictandStationIds();
        }

        for (int relevantStation : relevantStations) {
            int indexStation = forecast->GetStationRowFromId(relevantStation);

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
                int idx = iLead;
                if (timeShiftEndAccumulation) {
                    if (iLead == 0) {
                        continue;
                    }
                    idx -= 1;
                }
                if (isnan(maxValues[idx])) {
                    maxValues[idx] = -999999;
                }

                float thisVal = 0;

                // Get values
                a1f theseVals = forecast->GetAnalogsValuesRaw(iLead, indexStation);

                // Process quantiles
                if (asHasNaN(&theseVals[0], &theseVals[theseVals.size() - 1])) {
                    thisVal = NAN;
                } else {
                    float forecastVal = asGetValueForQuantile(theseVals, quantileThreshold);
                    forecastVal *= factor;
                    thisVal = forecastVal;
                }

                // Keep it if higher
                if (thisVal > maxValues[idx]) {
                    maxValues[idx] = thisVal;
                }
            }
        }
    }

    return maxValues;
}

a1f asResultsForecastAggregator::GetOverallMaxValues(a1f& dates, int returnPeriodRef, float quantileThreshold) const {
    a2f allMax = a2f::Zero(dates.size(), m_forecasts.size());

    for (int iMethod = 0; iMethod < (int)m_forecasts.size(); iMethod++) {
        a1f values = GetMethodMaxValues(dates, iMethod, returnPeriodRef, quantileThreshold);
        if (values.size() == dates.size()) {
            allMax.col(iMethod) = values;
        } else {
            int subDailySteps = values.size() / dates.size();
            for (int iDate = 0; iDate < dates.size(); ++iDate) {
                float maxVal = 0;
                for (int i = 0; i < subDailySteps; ++i) {
                    maxVal = wxMax(maxVal, values(iDate * subDailySteps + i));
                }
                allMax(iDate, iMethod) = maxVal;
            }
        }
    }

    // Remove NaNs
    allMax = (allMax.isFinite()).select(allMax, 0);

    // Extract the highest values
    a1f values = allMax.rowwise().maxCoeff();

    return values;
}

bool asResultsForecastAggregator::ExportSyntheticFullXml(const wxString& dirPath) const {
    // Quantile values
    a1f quantiles(3);
    quantiles << 20, 60, 90;

    // Create 1 file per method
    for (int methodRow = 0; methodRow < (int)m_forecasts.size(); methodRow++) {
        // Filename
        wxString filePath = dirPath;
        filePath.Append(DS);
        wxString dirStructure = "YYYY";
        dirStructure.Append(DS).Append("MM").Append(DS).Append("DD");
        wxString directory = asTime::GetStringTime(m_forecasts[methodRow][0]->GetLeadTimeOrigin(), dirStructure);
        filePath.Append(directory).Append(DS);
        wxString forecastName = m_forecasts[methodRow][0]->GetMethodId();
        wxString nowStr = asTime::GetStringTime(m_forecasts[methodRow][0]->GetLeadTimeOrigin(), "YYYY_MM_DD_hh");
        wxString ext = "xml";
        wxString filename = asStrF("%s.%s.%s", nowStr, forecastName, ext);
        filePath.Append(filename);

        // Create file
        asFileXml fileExport(filePath, asFile::Replace);
        if (!fileExport.Open()) return false;

        // General attributes
        fileExport.GetRoot()->AddAttribute("created", asTime::GetStringTime(asTime::NowMJD(), "DD.MM.YYYY HH"));

        // Method description
        auto nodeMethod = new wxXmlNode(wxXML_ELEMENT_NODE, "method");
        nodeMethod->AddChild(fileExport.CreateNode("id", m_forecasts[methodRow][0]->GetMethodId()));
        nodeMethod->AddChild(fileExport.CreateNode("name", m_forecasts[methodRow][0]->GetMethodIdDisplay()));
        nodeMethod->AddChild(fileExport.CreateNode("description", m_forecasts[methodRow][0]->GetDescription()));
        fileExport.AddChild(nodeMethod);

        // Reference axis
        if (m_forecasts[methodRow][0]->HasReferenceValues()) {
            a1f refAxis = m_forecasts[methodRow][0]->GetReferenceAxis();
            auto nodeReferenceAxis = new wxXmlNode(wxXML_ELEMENT_NODE, "reference_axis");
            for (float axis : refAxis) {
                nodeReferenceAxis->AddChild(fileExport.CreateNode("reference", asStrF("%.2f", axis)));
            }
            fileExport.AddChild(nodeReferenceAxis);
        }

        // Target dates
        a1f targetDates = m_forecasts[methodRow][0]->GetTargetDates();
        auto nodeTargetDates = new wxXmlNode(wxXML_ELEMENT_NODE, "target_dates");
        for (int i = 0; i < targetDates.size(); i++) {
            auto nodeTargetDate = new wxXmlNode(wxXML_ELEMENT_NODE, "target_date");
            nodeTargetDate->AddChild(
                fileExport.CreateNode("date", asTime::GetStringTime(targetDates[i], "DD.MM.YYYY HH")));
            nodeTargetDate->AddChild(
                fileExport.CreateNode("analogs_nb", m_forecasts[methodRow][0]->GetAnalogsNumber(i)));
            nodeTargetDates->AddChild(nodeTargetDate);
        }
        fileExport.AddChild(nodeTargetDates);

        // Quantiles
        auto nodeQuantiles = new wxXmlNode(wxXML_ELEMENT_NODE, "quantile_names");
        for (float quantile : quantiles) {
            nodeQuantiles->AddChild(fileExport.CreateNode("quantile", asStrF("%d", (int)quantile)));
        }
        fileExport.AddChild(nodeQuantiles);

        // Results per station
        a1i stationIds = m_forecasts[methodRow][0]->GetStationIds();
        a2f referenceValues;
        if (m_forecasts[methodRow][0]->HasReferenceValues()) {
            referenceValues = m_forecasts[methodRow][0]->GetReferenceValues();
            wxASSERT(referenceValues.rows() == stationIds.size());
        }
        auto nodeStations = new wxXmlNode(wxXML_ELEMENT_NODE, "stations");
        for (int i = 0; i < stationIds.size(); i++) {
            // Get specific forecast
            int forecastRow = GetForecastRowSpecificForStationId(methodRow, stationIds[i]);
            asResultsForecast* forecast = m_forecasts[methodRow][forecastRow];

            // Set station properties
            auto nodeStation = new wxXmlNode(wxXML_ELEMENT_NODE, "station");
            nodeStation->AddChild(fileExport.CreateNode("id", stationIds[i]));
            nodeStation->AddChild(fileExport.CreateNode("official_id", forecast->GetStationOfficialId(i)));
            nodeStation->AddChild(fileExport.CreateNode("name", forecast->GetStationName(i)));
            nodeStation->AddChild(fileExport.CreateNode("x", forecast->GetStationXCoord(i)));
            nodeStation->AddChild(fileExport.CreateNode("y", forecast->GetStationYCoord(i)));
            nodeStation->AddChild(fileExport.CreateNode("height", forecast->GetStationHeight(i)));
            nodeStation->AddChild(fileExport.CreateNode("specific_parameters", forecast->GetSpecificTagDisplay()));

            // Set reference values
            if (forecast->HasReferenceValues()) {
                auto nodeReferenceValues = new wxXmlNode(wxXML_ELEMENT_NODE, "reference_values");
                for (int j = 0; j < referenceValues.cols(); j++) {
                    nodeReferenceValues->AddChild(
                        fileExport.CreateNode("value", asStrF("%.2f", referenceValues(i, j))));
                }
                nodeStation->AddChild(nodeReferenceValues);
            }

            // Set 10 best analogs
            auto nodeBestAnalogs = new wxXmlNode(wxXML_ELEMENT_NODE, "best_analogs");
            for (int j = 0; j < targetDates.size(); j++) {
                a1f analogValues = forecast->GetAnalogsValuesRaw(j, i);
                a1f analogDates = forecast->GetAnalogsDates(j);
                a1f analogCriteria = forecast->GetAnalogsCriteria(j);
                wxASSERT(analogValues.size() == analogDates.size());
                wxASSERT(analogValues.size() == analogCriteria.size());

                auto nodeTargetDate = new wxXmlNode(wxXML_ELEMENT_NODE, "target_date");
                for (int k = 0; k < wxMin(10, analogValues.size()); k++) {
                    auto nodeAnalog = new wxXmlNode(wxXML_ELEMENT_NODE, "analog");
                    nodeAnalog->AddChild(
                        fileExport.CreateNode("date", asTime::GetStringTime(analogDates[k], "DD.MM.YYYY HH")));
                    nodeAnalog->AddChild(fileExport.CreateNode("value", asStrF("%.1f", analogValues[k])));
                    nodeAnalog->AddChild(fileExport.CreateNode("criteria", asStrF("%.1f", analogCriteria[k])));

                    nodeTargetDate->AddChild(nodeAnalog);
                }
                nodeBestAnalogs->AddChild(nodeTargetDate);
            }
            nodeStation->AddChild(nodeBestAnalogs);

            // Set quantiles
            auto nodeAnalogsQuantiles = new wxXmlNode(wxXML_ELEMENT_NODE, "analogs_quantiles");
            for (int j = 0; j < targetDates.size(); j++) {
                a1f analogValues = forecast->GetAnalogsValuesRaw(j, i);

                auto nodeTargetDate = new wxXmlNode(wxXML_ELEMENT_NODE, "target_date");
                for (int k = 0; k < wxMin(10, quantiles.size()); k++) {
                    float pcVal = asGetValueForQuantile(analogValues, quantiles[k] / 100);
                    nodeTargetDate->AddChild(fileExport.CreateNode("quantile", asStrF("%.1f", pcVal)));
                }
                nodeAnalogsQuantiles->AddChild(nodeTargetDate);
            }
            nodeStation->AddChild(nodeAnalogsQuantiles);

            // Set mean
            auto nodeAnalogsMean = new wxXmlNode(wxXML_ELEMENT_NODE, "analogs_mean");
            for (int j = 0; j < targetDates.size(); j++) {
                a1f analogValues = forecast->GetAnalogsValuesRaw(j, i);
                auto nodeTargetDate = new wxXmlNode(wxXML_ELEMENT_NODE, "target_date");
                float mean = analogValues.mean();
                nodeTargetDate->AddChild(fileExport.CreateNode("mean", asStrF("%.1f", mean)));
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

bool asResultsForecastAggregator::ExportSyntheticSmallCsv(const wxString& dirPath) const {
    // Quantile values
    a1f quantiles(6);
    quantiles << 0, 25, 50, 75, 90, 100;

    // Create 1 file per method
    for (int methodRow = 0; methodRow < (int)m_forecasts.size(); methodRow++) {
        // Filename
        wxString filePath = dirPath;
        filePath.Append(DS);
        wxString dirStructure = "YYYY";
        dirStructure.Append(DS).Append("MM").Append(DS).Append("DD");
        wxString directory = asTime::GetStringTime(m_forecasts[methodRow][0]->GetLeadTimeOrigin(), dirStructure);
        filePath.Append(directory).Append(DS);
        wxString forecastName = m_forecasts[methodRow][0]->GetMethodId();
        wxString nowStr = asTime::GetStringTime(m_forecasts[methodRow][0]->GetLeadTimeOrigin(), "YYYY_MM_DD_hh");
        wxString filename = asStrF("%s.%s.%s", nowStr, forecastName, "txt");
        filePath.Append(filename);

        // Create file
        asFileText fileExport(filePath, asFile::Replace);
        if (!fileExport.Open()) return false;

        // Method description
        fileExport.AddContent(asStrF("method: %s | %s\n", m_forecasts[methodRow][0]->GetMethodId(),
                                     m_forecasts[methodRow][0]->GetMethodIdDisplay()));

        // Results per station
        a1i stationIds = m_forecasts[methodRow][0]->GetStationIds();

        for (int i = 0; i < stationIds.size(); i++) {
            // Get specific forecast
            int forecastRow = GetForecastRowSpecificForStationId(methodRow, stationIds[i]);
            asResultsForecast* forecast = m_forecasts[methodRow][forecastRow];

            // Set station properties
            fileExport.AddContent(asStrF("station %d: %s (%s)\n", stationIds[i], forecast->GetStationName(i),
                                         forecast->GetSpecificTagDisplay()));
        }
        fileExport.AddContent("\n");

        // Quantiles
        wxString headerQuantiles;
        for (int k = 0; k < quantiles.size(); k++) {
            headerQuantiles.Append(asStrF("q%d raw; q%d transformed; ", (int)quantiles[k], (int)quantiles[k]));
        }

        // Headers
        fileExport.AddContent(
            asStrF("station; lead time; nb analogs; mean raw; mean transformed; %s\n", headerQuantiles));

        for (int i = 0; i < stationIds.size(); i++) {
            // Get specific forecast
            int forecastRow = GetForecastRowSpecificForStationId(methodRow, stationIds[i]);
            asResultsForecast* forecast = m_forecasts[methodRow][forecastRow];

            // Target dates
            a1f targetDates = forecast->GetTargetDates();
            for (int j = 0; j < targetDates.size(); j++) {
                a1f analogValuesRaw = forecast->GetAnalogsValuesRaw(j, i);
                a1f analogValuesNorm = forecast->GetAnalogsValuesNorm(j, i);

                wxString valuesQuantiles;
                for (int k = 0; k < quantiles.size(); k++) {
                    float pcValRaw = asGetValueForQuantile(analogValuesRaw, quantiles[k] / 100);
                    float pcValNorm = asGetValueForQuantile(analogValuesNorm, quantiles[k] / 100);
                    valuesQuantiles.Append(asStrF("%.2f; %.2f; ", pcValRaw, pcValNorm));
                }

                fileExport.AddContent(asStrF("%d; %s; %d; %.2f; %.2f; %s\n", stationIds[i],
                                             asTime::GetStringTime(targetDates[j], "DD.MM.YYYY HH"),
                                             forecast->GetAnalogsNumber(j), analogValuesRaw.mean(),
                                             analogValuesNorm.mean(), valuesQuantiles));
            }
        }

        fileExport.Close();
    }

    return true;
}

bool asResultsForecastAggregator::ExportSyntheticCustomCsvFVG(const wxString& dirPath) const {
    // Quantile values
    a1f quantiles(6);
    quantiles << 0, 25, 50, 75, 90, 100;

    // Create 1 file per method
    for (int methodRow = 0; methodRow < (int)m_forecasts.size(); methodRow++) {
        // Filename
        wxString filePath = dirPath;
        filePath.Append(DS);
        wxString dirStructure = "YYYYMMDD";
        wxString directory = asTime::GetStringTime(m_forecasts[methodRow][0]->GetLeadTimeOrigin(), dirStructure);
        filePath.Append(directory).Append(DS);
        wxString forecastName = m_forecasts[methodRow][0]->GetMethodId();
        wxString nowStr = asTime::GetStringTime(m_forecasts[methodRow][0]->GetLeadTimeOrigin(), "YYYY_MM_DD_hh");
        wxString filename = asStrF("%s.%s.%s", nowStr, forecastName, "txt");
        filePath.Append(filename);

        // Create file
        asFileText fileExport(filePath, asFile::Replace);
        if (!fileExport.Open()) return false;

        // Results per station
        a1i stationIds = m_forecasts[methodRow][0]->GetStationIds();

        // Quantiles
        wxString headerQuantiles;
        for (int k = 0; k < quantiles.size(); k++) {
            headerQuantiles.Append(asStrF("q%d; ", (int)quantiles[k]));
        }

        // Headers
        fileExport.AddContent(
            asStrF("area; lead time; forecast; nb analogs; mean raw; mean transformed; %s\n", headerQuantiles));

        for (int i = 0; i < stationIds.size(); i++) {
            // Get specific forecast
            int forecastRow = GetForecastRowSpecificForStationId(methodRow, stationIds[i]);
            asResultsForecast* forecast = m_forecasts[methodRow][forecastRow];

            // Target dates
            a1f targetDates = forecast->GetTargetDates();
            for (int j = 0; j < targetDates.size(); j++) {
                a1f analogValuesRaw = forecast->GetAnalogsValuesRaw(j, i);
                a1f analogValuesNorm = forecast->GetAnalogsValuesNorm(j, i);

                wxString valuesQuantiles;
                for (int k = 0; k < quantiles.size(); k++) {
                    float pcValNorm = asGetValueForQuantile(analogValuesNorm, quantiles[k] / 100);
                    if (isnan(pcValNorm)) {
                        valuesQuantiles.Append("-9999; ");
                        continue;
                    }
                    valuesQuantiles.Append(asStrF("%.2f; ", pcValNorm));
                }

                int hourForecast = int(24 * (targetDates[j] - forecast->GetLeadTimeOrigin()));
                float meanRaw = analogValuesRaw.mean();
                float meanNorm = analogValuesNorm.mean();
                if (isnan(meanRaw) || isnan(meanNorm)) {
                    fileExport.AddContent(asStrF("%d; %s; %02d; %d; -9999; -9999; %s\n", stationIds[i] - 2,
                                                 asTime::GetStringTime(targetDates[j], "YYYYMMDDHH"), hourForecast,
                                                 forecast->GetAnalogsNumber(j), valuesQuantiles));
                    continue;
                }

                fileExport.AddContent(asStrF("%d; %s; %02d; %d; %.2f; %.2f; %s\n", stationIds[i] - 2,
                                             asTime::GetStringTime(targetDates[j], "YYYYMMDDHH"), hourForecast,
                                             forecast->GetAnalogsNumber(j), meanRaw, meanNorm, valuesQuantiles));
            }
        }

        fileExport.Close();
    }

    return true;
}
