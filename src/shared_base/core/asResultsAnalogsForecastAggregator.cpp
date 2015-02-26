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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2015 Pascal Horton, Terr@num.
 */

#include "asResultsAnalogsForecastAggregator.h"
#include "asFileXml.h"


asResultsAnalogsForecastAggregator::asResultsAnalogsForecastAggregator()
:
wxObject()
{
    
}

asResultsAnalogsForecastAggregator::~asResultsAnalogsForecastAggregator()
{
    ClearArrays();
}

void asResultsAnalogsForecastAggregator::Add(asResultsAnalogsForecast* forecast)
{
    bool compatible = true;
    bool createNewMethodRow = true;

    for (int methodRow=0; methodRow<m_Forecasts.size(); methodRow++)
    {
        wxASSERT(m_Forecasts[methodRow].size()>0);
        asResultsAnalogsForecast* refForecast = m_Forecasts[methodRow][0];

        if (!refForecast->GetMethodId().IsSameAs(forecast->GetMethodId(), false)) compatible = false;
        if (refForecast->GetPredictandParameter() != forecast->GetPredictandParameter()) compatible = false;
        if (refForecast->GetPredictandTemporalResolution() != forecast->GetPredictandTemporalResolution()) compatible = false;
        if (refForecast->GetPredictandSpatialAggregation() != forecast->GetPredictandSpatialAggregation()) compatible = false;
        if (!refForecast->GetPredictandDatasetId().IsSameAs(forecast->GetPredictandDatasetId(), false)) compatible = false;
        if (!refForecast->GetPredictandDatabase().IsSameAs(forecast->GetPredictandDatabase(), false)) compatible = false;

        if (compatible)
        {
            // Detailed checks
            if (forecast->IsCompatibleWith(refForecast))
            {
                m_Forecasts[methodRow].push_back(forecast);
                m_PastForecasts[methodRow].resize(m_Forecasts[methodRow].size());
                createNewMethodRow = false;
                break;
            }
            else
            {
                asLogError(wxString::Format(_("The forecast \"%s\" (%s) is not fully compatible with \"%s\" (%s)"), 
                    forecast->GetSpecificTagDisplay().c_str(), forecast->GetMethodIdDisplay().c_str(),
                    refForecast->GetSpecificTagDisplay().c_str(), refForecast->GetMethodIdDisplay().c_str()));
            }
        }
    }

    if (createNewMethodRow)
    {
        m_Forecasts.resize(m_Forecasts.size()+1);
        m_PastForecasts.resize(m_PastForecasts.size()+1);
        m_Forecasts[m_Forecasts.size()-1].push_back(forecast);
        m_PastForecasts[m_PastForecasts.size()-1].resize(1);
    }
}

void asResultsAnalogsForecastAggregator::AddPastForecast(int methodRow, int forecastRow, asResultsAnalogsForecast* forecast)
{
    bool compatible = true;

    wxASSERT(m_Forecasts.size()>methodRow);
    wxASSERT(m_PastForecasts.size()>methodRow);
    wxASSERT(m_Forecasts[methodRow].size()>forecastRow);
    wxASSERT(m_PastForecasts[methodRow].size()>forecastRow);

    asResultsAnalogsForecast* refForecast = m_Forecasts[methodRow][forecastRow];

    if (!refForecast->GetMethodId().IsSameAs(forecast->GetMethodId(), false)) compatible = false;
    if (!refForecast->GetSpecificTag().IsSameAs(forecast->GetSpecificTag(), false)) compatible = false;
    if (refForecast->GetPredictandParameter() != forecast->GetPredictandParameter()) compatible = false;
    if (refForecast->GetPredictandTemporalResolution() != forecast->GetPredictandTemporalResolution()) compatible = false;
    if (refForecast->GetPredictandSpatialAggregation() != forecast->GetPredictandSpatialAggregation()) compatible = false;
    if (!refForecast->GetPredictandDatasetId().IsSameAs(forecast->GetPredictandDatasetId(), false)) compatible = false;
    if (!refForecast->GetPredictandDatabase().IsSameAs(forecast->GetPredictandDatabase(), false)) compatible = false;

    if (compatible)
    {
        m_PastForecasts[methodRow][forecastRow].push_back(forecast);
    }
    else
    {
        asLogError(wxString::Format(_("The past forecast \"%s\" (%s) is not fully compatible with the current version of \"%s\" (%s)"), 
            forecast->GetSpecificTagDisplay().c_str(), forecast->GetMethodIdDisplay().c_str(),
            refForecast->GetSpecificTagDisplay().c_str(), refForecast->GetMethodIdDisplay().c_str()));
    }
}

void asResultsAnalogsForecastAggregator::ClearArrays()
{
    for (int i=0; (unsigned)i<m_Forecasts.size(); i++)
    {
        for (int j=0; (unsigned)j<m_Forecasts[i].size(); j++)
        {
            wxDELETE(m_Forecasts[i][j]);
        }
    }
    m_Forecasts.clear();
    
    for (int i=0; (unsigned)i<m_PastForecasts.size(); i++)
    {
        for (int j=0; (unsigned)j<m_PastForecasts[i].size(); j++)
        {
            for (int k=0; (unsigned)k<m_PastForecasts[i][j].size(); k++)
            {
                wxDELETE(m_PastForecasts[i][j][k]);
            }
        }
    }
    m_PastForecasts.clear();
}

int asResultsAnalogsForecastAggregator::GetMethodsNb()
{
    return (int)m_Forecasts.size();
}

int asResultsAnalogsForecastAggregator::GetForecastsNb(int methodRow)
{
    wxASSERT(m_Forecasts.size()>methodRow);
    return (int)m_Forecasts[methodRow].size();
}

int asResultsAnalogsForecastAggregator::GetPastMethodsNb()
{
    return (int)m_PastForecasts.size();
}

int asResultsAnalogsForecastAggregator::GetPastForecastsNb(int methodRow)
{
    wxASSERT(m_PastForecasts.size()>methodRow);
    return (int)m_PastForecasts[methodRow].size();
}

int asResultsAnalogsForecastAggregator::GetPastForecastsNb(int methodRow, int forecastRow)
{
    wxASSERT(m_PastForecasts.size()>(unsigned)methodRow);
    wxASSERT(m_PastForecasts[methodRow].size()>(unsigned)forecastRow);
    return (int)m_PastForecasts[methodRow][forecastRow].size();
}

asResultsAnalogsForecast* asResultsAnalogsForecastAggregator::GetForecast(int methodRow, int forecastRow)
{
    wxASSERT(m_Forecasts.size()>(unsigned)methodRow);
    wxASSERT(m_Forecasts[methodRow].size()>(unsigned)forecastRow);
    return m_Forecasts[methodRow][forecastRow];
}

asResultsAnalogsForecast* asResultsAnalogsForecastAggregator::GetPastForecast(int methodRow, int forecastRow, int leadtimeRow)
{
    wxASSERT(m_PastForecasts.size()>(unsigned)methodRow);
    wxASSERT(m_PastForecasts[methodRow].size()>(unsigned)forecastRow);
    return m_PastForecasts[methodRow][forecastRow][leadtimeRow];
}

wxString asResultsAnalogsForecastAggregator::GetForecastName(int methodRow, int forecastRow)
{
    wxString name = wxEmptyString;

    if (m_Forecasts.size()==0) return wxEmptyString;

    wxASSERT(m_Forecasts.size()>(unsigned)methodRow);

    if(m_Forecasts.size()>(unsigned)methodRow && m_Forecasts[methodRow].size()>(unsigned)forecastRow)
    {
        name = m_Forecasts[methodRow][forecastRow]->GetMethodIdDisplay();

        if (!name.IsSameAs(m_Forecasts[methodRow][forecastRow]->GetMethodId()))
        {
            name.Append(wxString::Format(" (%s)", m_Forecasts[methodRow][forecastRow]->GetMethodId().c_str()));
        }

        if (!m_Forecasts[methodRow][forecastRow]->GetSpecificTag().IsEmpty())
        {
            name.Append(" - ");
            name.Append(m_Forecasts[methodRow][forecastRow]->GetSpecificTagDisplay());
        }
    }

    wxASSERT(!name.IsEmpty());

    return name;
}

wxString asResultsAnalogsForecastAggregator::GetMethodName(int methodRow)
{
    wxString name = wxEmptyString;

    if (m_Forecasts.size()==0) return wxEmptyString;

    wxASSERT(m_Forecasts.size()>(unsigned)methodRow);

    if(m_Forecasts.size()>(unsigned)methodRow && m_Forecasts[methodRow].size()>0)
    {
        name = m_Forecasts[methodRow][0]->GetMethodIdDisplay();

        if (!name.IsSameAs(m_Forecasts[methodRow][0]->GetMethodId()))
        {
            name.Append(wxString::Format(" (%s)", m_Forecasts[methodRow][0]->GetMethodId().c_str()));
        }
    }

    wxASSERT(!name.IsEmpty());

    return name;
}

VectorString asResultsAnalogsForecastAggregator::GetAllMethodIds()
{
    VectorString methodsIds;

    for (int methodRow=1; methodRow<m_Forecasts.size(); methodRow++)
    {
        wxASSERT(m_Forecasts[methodRow].size()>0);
        methodsIds.push_back(m_Forecasts[methodRow][0]->GetMethodId());
    }

    return methodsIds;
}

VectorString asResultsAnalogsForecastAggregator::GetAllMethodNames()
{
    VectorString names;

    for (unsigned int methodRow=0; methodRow<m_Forecasts.size(); methodRow++)
    {
        wxASSERT(m_Forecasts[methodRow].size()>0);

        wxString methodName = m_Forecasts[methodRow][0]->GetMethodIdDisplay();
        if (!methodName.IsSameAs(m_Forecasts[methodRow][0]->GetMethodId()))
        {
            methodName.Append(wxString::Format(" (%s)", m_Forecasts[methodRow][0]->GetMethodId().c_str()));
        }
        names.push_back(methodName);
    }

    wxASSERT(names.size()>0);

    return names;
}

VectorString asResultsAnalogsForecastAggregator::GetAllForecastNames()
{
    VectorString names;

    for (unsigned int methodRow=0; methodRow<m_Forecasts.size(); methodRow++)
    {
        wxASSERT(m_Forecasts[methodRow].size()>0);

        wxString methodName = m_Forecasts[methodRow][0]->GetMethodIdDisplay();
        if (!methodName.IsSameAs(m_Forecasts[methodRow][0]->GetMethodId()))
        {
            methodName.Append(wxString::Format(" (%s)", m_Forecasts[methodRow][0]->GetMethodId().c_str()));
        }

        for (unsigned int forecastRow=0; forecastRow<m_Forecasts[methodRow].size(); forecastRow++)
        {
            wxString name = methodName;

            if (!m_Forecasts[methodRow][forecastRow]->GetSpecificTag().IsEmpty())
            {
                name.Append(" - ");
                name.Append(m_Forecasts[methodRow][forecastRow]->GetSpecificTagDisplay());
            }
            names.push_back(name);
        }
    }

    wxASSERT(names.size()>0);

    return names;
}

wxArrayString asResultsAnalogsForecastAggregator::GetAllForecastNamesWxArray()
{
    wxArrayString names;

    for (unsigned int methodRow=0; methodRow<m_Forecasts.size(); methodRow++)
    {
        wxASSERT(m_Forecasts[methodRow].size()>0);

        wxString methodName = m_Forecasts[methodRow][0]->GetMethodIdDisplay();
        if (!methodName.IsSameAs(m_Forecasts[methodRow][0]->GetMethodId()))
        {
            methodName.Append(wxString::Format(" (%s)", m_Forecasts[methodRow][0]->GetMethodId().c_str()));
        }

        for (unsigned int forecastRow=0; forecastRow<m_Forecasts[methodRow].size(); forecastRow++)
        {
            wxString name = methodName;

            if (!m_Forecasts[methodRow][forecastRow]->GetSpecificTag().IsEmpty())
            {
                name.Append(" - ");
                name.Append(m_Forecasts[methodRow][forecastRow]->GetSpecificTagDisplay());
            }
            names.Add(name);
        }
    }

    wxASSERT(names.size()>0);

    return names;
}

VectorString asResultsAnalogsForecastAggregator::GetFilePaths()
{
    VectorString files;

    for (int i=0; (unsigned)i<m_Forecasts.size(); i++)
    {
        for (int j=0; (unsigned)j<m_Forecasts[i].size(); j++)
        {
            files.push_back(m_Forecasts[i][j]->GetFilePath());
        }
    }

    return files;
}

wxString asResultsAnalogsForecastAggregator::GetFilePath(int methodRow, int forecastRow)
{
    if (forecastRow<0)
    {
        forecastRow = 0;
    }

    return m_Forecasts[methodRow][forecastRow]->GetFilePath();
}

wxArrayString asResultsAnalogsForecastAggregator::GetFilePathsWxArray()
{
    wxArrayString files;

    for (int i=0; (unsigned)i<m_Forecasts.size(); i++)
    {
        for (int j=0; (unsigned)j<m_Forecasts[i].size(); j++)
        {
            files.Add(m_Forecasts[i][j]->GetFilePath());
        }
    }

    return files;
}

Array1DFloat asResultsAnalogsForecastAggregator::GetTargetDates(int methodRow)
{
    double firstDate = 9999999999, lastDate = 0;

    for (unsigned int forecastRow=0; forecastRow<m_Forecasts[methodRow].size(); forecastRow++)
    {
        Array1DFloat fcastDates = m_Forecasts[methodRow][forecastRow]->GetTargetDates();
        if (fcastDates[0]<firstDate)
        {
            firstDate = fcastDates[0];
        }
        if (fcastDates[fcastDates.size()-1]>lastDate)
        {
            lastDate = fcastDates[fcastDates.size()-1];
        }
    }

    int size = asTools::Round(lastDate-firstDate+1);
    Array1DFloat dates = Array1DFloat::LinSpaced(size,firstDate,lastDate);

    return dates;
}

Array1DFloat asResultsAnalogsForecastAggregator::GetTargetDates(int methodRow, int forecastRow)
{
    return m_Forecasts[methodRow][forecastRow]->GetTargetDates();
}

Array1DFloat asResultsAnalogsForecastAggregator::GetFullTargetDates()
{
    double firstDate = 9999999999, lastDate = 0;

    for (unsigned int methodRow=0; methodRow<m_Forecasts.size(); methodRow++)
    {
        for (unsigned int forecastRow=0; forecastRow<m_Forecasts[methodRow].size(); forecastRow++)
        {
            Array1DFloat fcastDates = m_Forecasts[methodRow][forecastRow]->GetTargetDates();
            if (fcastDates[0]<firstDate)
            {
                firstDate = fcastDates[0];
            }
            if (fcastDates[fcastDates.size()-1]>lastDate)
            {
                lastDate = fcastDates[fcastDates.size()-1];
            }
        }
    }

    int size = asTools::Round(lastDate-firstDate+1);
    Array1DFloat dates = Array1DFloat::LinSpaced(size,firstDate,lastDate);

    return dates;
}

int asResultsAnalogsForecastAggregator::GetForecastRowSpecificForStationId(int methodRow, int stationId)
{
    // Pick up the most relevant forecast for the station
    for (int i=0; i<GetForecastsNb(methodRow); i++)
    {
        asResultsAnalogsForecast* forecast = m_Forecasts[methodRow][i];
        if (forecast->IsSpecificForStationId(stationId))
        {
            return i;
        }
    }

    asLogWarning(wxString::Format(_("No specific forecast was found for station ID %d"), stationId));

    return 0;
}

int asResultsAnalogsForecastAggregator::GetForecastRowSpecificForStationRow(int methodRow, int stationRow)
{
    // Pick up the most relevant forecast for the station
    for (int i=0; i<GetForecastsNb(methodRow); i++)
    {
        asResultsAnalogsForecast* forecast = m_Forecasts[methodRow][i];
        int stationId = forecast->GetStationId(stationRow);
        if (forecast->IsSpecificForStationId(stationId))
        {
            return i;
        }
    }

    asLogWarning(wxString::Format(_("No specific forecast was found for station n°%d"), stationRow));

    return 0;
}

wxArrayString asResultsAnalogsForecastAggregator::GetStationNames(int methodRow, int forecastRow)
{
    wxArrayString stationNames;

    if (m_Forecasts.size()==0) return stationNames;

    wxASSERT(m_Forecasts.size()>(unsigned)methodRow);
    wxASSERT(m_Forecasts[methodRow].size()>(unsigned)forecastRow);

    stationNames = m_Forecasts[methodRow][forecastRow]->GetStationNamesWxArrayString();

    return stationNames;
}

wxString asResultsAnalogsForecastAggregator::GetStationName(int methodRow, int forecastRow, int stationRow)
{
    wxString stationName;

    if (m_Forecasts.size()==0) return wxEmptyString;
    
    wxASSERT(m_Forecasts.size()>(unsigned)methodRow);
    wxASSERT(m_Forecasts[methodRow].size()>(unsigned)forecastRow);

    stationName = m_Forecasts[methodRow][forecastRow]->GetStationName(stationRow);

    return stationName;
}

wxArrayString asResultsAnalogsForecastAggregator::GetStationNamesWithHeights(int methodRow, int forecastRow)
{
    wxArrayString stationNames;

    if (m_Forecasts.size()==0) return stationNames;
    
    wxASSERT(m_Forecasts.size()>(unsigned)methodRow);
    wxASSERT(m_Forecasts[methodRow].size()>(unsigned)forecastRow);

    stationNames = m_Forecasts[methodRow][forecastRow]->GetStationNamesAndHeightsWxArrayString();

    return stationNames;
}

wxString asResultsAnalogsForecastAggregator::GetStationNameWithHeight(int methodRow, int forecastRow, int stationRow)
{
    wxString stationName;

    if (m_Forecasts.size()==0) return wxEmptyString;
    
    wxASSERT(m_Forecasts.size()>(unsigned)methodRow);
    wxASSERT(m_Forecasts[methodRow].size()>(unsigned)forecastRow);

    stationName = m_Forecasts[methodRow][forecastRow]->GetStationNameAndHeight(stationRow);

    return stationName;
}

int asResultsAnalogsForecastAggregator::GetLeadTimeLength(int methodRow, int forecastRow)
{
    if (m_Forecasts.size()==0) return 0;
    
    wxASSERT(m_Forecasts.size()>(unsigned)methodRow);
    wxASSERT(m_Forecasts[methodRow].size()>(unsigned)forecastRow);

    int length = m_Forecasts[methodRow][forecastRow]->GetTargetDatesLength();

    wxASSERT(length>0);

    return length;
}

int asResultsAnalogsForecastAggregator::GetLeadTimeLengthMax()
{
    if (m_Forecasts.size()==0) return 0;
    
    int length = 0;

    for (int i=0; i<m_Forecasts.size(); i++)
    {
        for (int j=0; j<m_Forecasts.size(); j++)
        {
            length = wxMax(length, m_Forecasts[i][j]->GetTargetDatesLength());
        }
    }

    return length;
}

wxArrayString asResultsAnalogsForecastAggregator::GetLeadTimes(int methodRow, int forecastRow)
{
    wxArrayString leadTimes;

    if (m_Forecasts.size()==0) return leadTimes;
    
    wxASSERT(m_Forecasts.size()>(unsigned)methodRow);
    wxASSERT(m_Forecasts[methodRow].size()>(unsigned)forecastRow);

    Array1DFloat dates = m_Forecasts[methodRow][forecastRow]->GetTargetDates();

    for (int i=0; i<dates.size(); i++)
    {
        leadTimes.Add(asTime::GetStringTime(dates[i], "DD.MM.YYYY"));
    }

    return leadTimes;
}

Array1DFloat asResultsAnalogsForecastAggregator::GetMethodMaxValues(Array1DFloat &dates, int methodRow, int returnPeriodRef, float percentileThreshold)
{
    wxASSERT(returnPeriodRef>=2);
    wxASSERT(percentileThreshold>0);
    wxASSERT(percentileThreshold<1);
    if (returnPeriodRef<2) returnPeriodRef = 2;
    if (percentileThreshold<=0) percentileThreshold = (float)0.9;
    if (percentileThreshold>1) percentileThreshold = (float)0.9;

    wxASSERT(m_Forecasts.size()>methodRow);

    Array1DFloat maxValues = Array1DFloat::Ones(dates.size());
    maxValues *= NaNFloat;

    for (int forecastRow=0; forecastRow<m_Forecasts[methodRow].size(); forecastRow++)
    {
        asResultsAnalogsForecast* forecast = m_Forecasts[methodRow][forecastRow];

        // Get return period index
        int indexReferenceAxis = asNOT_FOUND;
        if(forecast->HasReferenceValues())
        {
            Array1DFloat forecastReferenceAxis = forecast->GetReferenceAxis();
            indexReferenceAxis = asTools::SortedArraySearch(&forecastReferenceAxis[0], &forecastReferenceAxis[forecastReferenceAxis.size()-1], returnPeriodRef);
            if ( (indexReferenceAxis==asNOT_FOUND) || (indexReferenceAxis==asOUT_OF_RANGE) )
            {
                asLogError(_("The desired return period is not available in the forecast file."));
            }
        }

        // Check lead times effectively available for the current forecast
        int leadtimeMin = 0;
        int leadtimeMax = dates.size()-1;

        Array1DFloat availableDates = forecast->GetTargetDates();

        while (dates[leadtimeMin]<availableDates[0])
        {
            leadtimeMin++;
        }
        while (dates[leadtimeMax]>availableDates[availableDates.size()-1])
        {
            leadtimeMax--;
        }
        wxASSERT(leadtimeMin<leadtimeMax);

        // Get the values of the relevant stations only
        VectorInt relevantStations = forecast->GetPredictandStationIds();
        for (int i_st=0; i_st<relevantStations.size(); i_st++)
        {
            int indexStation = forecast->GetStationRowFromId(relevantStations[i_st]);
            
            // Get values for return period
            float factor = 1;
            if(forecast->HasReferenceValues())
            {
                float precip = forecast->GetReferenceValue(indexStation, indexReferenceAxis);
                wxASSERT(precip>0);
                wxASSERT(precip<500);
                factor = 1.0/precip;
                wxASSERT(factor>0);
            }

            for (int i_leadtime=leadtimeMin; i_leadtime<=leadtimeMax; i_leadtime++)
            {
                if (asTools::IsNaN(maxValues[i_leadtime]))
                {
                    maxValues[i_leadtime] = -999999;
                }

                float thisVal = 0;

                // Get values
                Array1DFloat theseVals = forecast->GetAnalogsValuesGross(i_leadtime, indexStation);

                // Process percentiles
                if(asTools::HasNaN(&theseVals[0], &theseVals[theseVals.size()-1]))
                {
                    thisVal = NaNFloat;
                }
                else
                {
                    float forecastVal = asTools::Percentile(&theseVals[0], &theseVals[theseVals.size()-1], percentileThreshold);
                    forecastVal *= factor;
                    thisVal = forecastVal;
                }

                // Keep it if higher
                if (thisVal>maxValues[i_leadtime])
                {
                    maxValues[i_leadtime] = thisVal;
                }
            }
        }
    }

    return maxValues;
}

Array1DFloat asResultsAnalogsForecastAggregator::GetOverallMaxValues(Array1DFloat &dates, int returnPeriodRef, float percentileThreshold)
{
    Array2DFloat allMax(dates.size(), m_Forecasts.size());

    for (int methodRow=0; methodRow<m_Forecasts.size(); methodRow++)
    {
        allMax.col(methodRow) = GetMethodMaxValues(dates, methodRow, returnPeriodRef, percentileThreshold);
    }

    // Extract the highest values
    Array1DFloat values = allMax.rowwise().maxCoeff();

    return values;
}

bool asResultsAnalogsForecastAggregator::ExportSyntheticXml()
{
    // Quantile values
    Array1DFloat quantiles(3);
    quantiles << 20, 60, 90;

    // Create 1 file per method
    for (int methodRow=0; methodRow<m_Forecasts.size(); methodRow++)
    {
        
        
        
        wxString filePath = wxString::Format("C:\\Users\\Pascal\\Desktop\\tets atmoswing\\export_%d.xml",methodRow);



       

        // Create file
        asFileXml fileExport(filePath, asFile::Replace);
        if(!fileExport.Open()) return false;

        // General attributes
        fileExport.GetRoot()->AddAttribute("created", asTime::GetStringTime(asTime::NowMJD(), "DD.MM.YYYY"));

        // Method description
        wxXmlNode * nodeMethod = new wxXmlNode(wxXML_ELEMENT_NODE ,"method" );
        nodeMethod->AddChild(fileExport.CreateNodeWithValue("id", m_Forecasts[methodRow][0]->GetMethodId()));
        nodeMethod->AddChild(fileExport.CreateNodeWithValue("name", m_Forecasts[methodRow][0]->GetMethodIdDisplay()));
        nodeMethod->AddChild(fileExport.CreateNodeWithValue("description", m_Forecasts[methodRow][0]->GetDescription()));
        fileExport.AddChild(nodeMethod);

        // Reference axis
        if (m_Forecasts[methodRow][0]->HasReferenceValues())
        {
            Array1DFloat refAxis = m_Forecasts[methodRow][0]->GetReferenceAxis();
            wxXmlNode * nodeReferenceAxis = new wxXmlNode(wxXML_ELEMENT_NODE ,"reference_axis" );
            for (int i=0; i<refAxis.size(); i++)
            {
                nodeReferenceAxis->AddChild(fileExport.CreateNodeWithValue("reference", wxString::Format("%.2f", refAxis[i])));
            }
            fileExport.AddChild(nodeReferenceAxis);
        }
        
        // Target dates
        Array1DFloat targetDates = m_Forecasts[methodRow][0]->GetTargetDates();
        wxXmlNode * nodeTargetDates = new wxXmlNode(wxXML_ELEMENT_NODE ,"target_dates" );
        for (int i=0; i<targetDates.size(); i++)
        {
            wxXmlNode * nodeTargetDate = new wxXmlNode(wxXML_ELEMENT_NODE ,"target_date" );
            nodeTargetDate->AddChild(fileExport.CreateNodeWithValue("date", asTime::GetStringTime(targetDates[i], "DD.MM.YYYY")));
            nodeTargetDate->AddChild(fileExport.CreateNodeWithValue("analogs_nb", m_Forecasts[methodRow][0]->GetAnalogsNumber(i)));
            nodeTargetDates->AddChild(nodeTargetDate);
        }
        fileExport.AddChild(nodeTargetDates);
        
        // Quantiles
        wxXmlNode * nodeQuantiles = new wxXmlNode(wxXML_ELEMENT_NODE ,"quantile_names" );
        for (int i=0; i<quantiles.size(); i++)
        {
            nodeQuantiles->AddChild(fileExport.CreateNodeWithValue("quantile", wxString::Format("%d", (int)quantiles[i])));
        }
        fileExport.AddChild(nodeQuantiles);

        // Results per station
        Array1DInt stationIds = m_Forecasts[methodRow][0]->GetStationIds();
        Array2DFloat referenceValues = m_Forecasts[methodRow][0]->GetReferenceValues();
        wxASSERT(referenceValues.rows()==stationIds.size());
        wxXmlNode * nodeStations = new wxXmlNode(wxXML_ELEMENT_NODE ,"stations" );
        for (int i=0; i<stationIds.size(); i++)
        {
            // Get specific forecast
            int forecastRow = GetForecastRowSpecificForStationId(methodRow, stationIds[i]);
            asResultsAnalogsForecast* forecast = m_Forecasts[methodRow][forecastRow];

            // Set station properties
            wxXmlNode * nodeStation = new wxXmlNode(wxXML_ELEMENT_NODE ,"station" );
            nodeStation->AddChild(fileExport.CreateNodeWithValue("id", stationIds[i]));
            nodeStation->AddChild(fileExport.CreateNodeWithValue("official_id", forecast->GetStationOfficialId(i)));
            nodeStation->AddChild(fileExport.CreateNodeWithValue("name", forecast->GetStationName(i)));
            nodeStation->AddChild(fileExport.CreateNodeWithValue("x", forecast->GetStationXCoord(i)));
            nodeStation->AddChild(fileExport.CreateNodeWithValue("y", forecast->GetStationYCoord(i)));
            nodeStation->AddChild(fileExport.CreateNodeWithValue("height", forecast->GetStationHeight(i)));
            nodeStation->AddChild(fileExport.CreateNodeWithValue("specific_parameters", forecast->GetSpecificTagDisplay()));

            // Set reference values
            if (forecast->HasReferenceValues())
            {
                wxXmlNode * nodeReferenceValues = new wxXmlNode(wxXML_ELEMENT_NODE ,"reference_values" );
                for (int j=0; j<referenceValues.cols(); j++)
                {
                    nodeReferenceValues->AddChild(fileExport.CreateNodeWithValue("value", wxString::Format("%.2f", referenceValues(i,j))));
                }
                nodeStation->AddChild(nodeReferenceValues);
            }

            // Set 10 best analogs
            wxXmlNode * nodeBestAnalogs = new wxXmlNode(wxXML_ELEMENT_NODE ,"best_analogs" );
            for (int j=0; j<targetDates.size(); j++)
            {
                Array1DFloat analogValues = forecast->GetAnalogsValuesGross(j, i);
                Array1DFloat analogDates = forecast->GetAnalogsDates(j);
                Array1DFloat analogCriteria = forecast->GetAnalogsCriteria(j);
                wxASSERT(analogValues.size()==analogDates.size());
                wxASSERT(analogValues.size()==analogCriteria.size());

                wxXmlNode * nodeTargetDate = new wxXmlNode(wxXML_ELEMENT_NODE ,"target_date" );
                for (int k=0; k<wxMin(10, analogValues.size()); k++)
                {
                    wxXmlNode * nodeAnalog = new wxXmlNode(wxXML_ELEMENT_NODE ,"analog" );
                    nodeAnalog->AddChild(fileExport.CreateNodeWithValue("date", asTime::GetStringTime(analogDates[k], "DD.MM.YYYY")));
                    nodeAnalog->AddChild(fileExport.CreateNodeWithValue("value", wxString::Format("%.1f", analogValues[k])));
                    nodeAnalog->AddChild(fileExport.CreateNodeWithValue("criteria", wxString::Format("%.1f", analogCriteria[k])));

                    nodeTargetDate->AddChild(nodeAnalog);
                }
                nodeBestAnalogs->AddChild(nodeTargetDate);
            }
            nodeStation->AddChild(nodeBestAnalogs);

            // Set quantiles
            wxXmlNode * nodeAnalogsQuantiles = new wxXmlNode(wxXML_ELEMENT_NODE ,"analogs_quantiles" );
            for (int j=0; j<targetDates.size(); j++)
            {
                Array1DFloat analogValues = forecast->GetAnalogsValuesGross(j, i);

                wxXmlNode * nodeTargetDate = new wxXmlNode(wxXML_ELEMENT_NODE ,"target_date" );
                for (int k=0; k<wxMin(10, quantiles.size()); k++)
                {
                    float pcVal = asTools::Percentile(&analogValues[0], &analogValues[analogValues.size()-1], quantiles[k]/100);
                    nodeTargetDate->AddChild(fileExport.CreateNodeWithValue("quantile", wxString::Format("%.1f", pcVal)));
                }
                nodeAnalogsQuantiles->AddChild(nodeTargetDate);
            }
            nodeStation->AddChild(nodeAnalogsQuantiles);

            nodeStations->AddChild(nodeStation);
        }
        fileExport.AddChild(nodeStations);

        fileExport.Save();
    }

    return true;
}