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


asResultsAnalogsForecastAggregator::asResultsAnalogsForecastAggregator()
:
wxObject()
{
    
}

asResultsAnalogsForecastAggregator::~asResultsAnalogsForecastAggregator()
{
    //dtor
}

Array1DFloat asResultsAnalogsForecastAggregator::GetMaxValues(Array1DFloat &dates, std::vector <asResultsAnalogsForecast*> forecasts, int returnPeriodRef, float percentileThreshold)
{
    wxASSERT(returnPeriodRef>=2);
    wxASSERT(percentileThreshold>0);
    wxASSERT(percentileThreshold<1);
    if (returnPeriodRef<2) returnPeriodRef = 2;
    if (percentileThreshold<=0) percentileThreshold = (float)0.9;
    if (percentileThreshold>1) percentileThreshold = (float)0.9;

    VectorString methodIds = ExtractMethodIds(forecasts);



    methodIds




    Array2DFloat allValues = Array2DFloat::Ones(forecasts.size(), dates.size());
    allValues *= NaNFloat;

    for (unsigned int i_model=0; i_model<forecasts.size(); i_model++)
    {
        asResultsAnalogsForecast* forecast = forecasts[i_model];
        int stationsNb = forecast->GetStationsNb();

        // Get return period index
        int indexReferenceAxis=asNOT_FOUND;
        if(forecast->HasReferenceValues())
        {
            Array1DFloat forecastReferenceAxis = forecast->GetReferenceAxis();

            indexReferenceAxis = asTools::SortedArraySearch(&forecastReferenceAxis[0], &forecastReferenceAxis[forecastReferenceAxis.size()-1], returnPeriodRef);
            if ( (indexReferenceAxis==asNOT_FOUND) || (indexReferenceAxis==asOUT_OF_RANGE) )
            {
                asLogError(_("The desired return period is not available in the forecast file."));
            }
        }

        // Check lead times effectively available for the current model
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

        for (int i_leadtime=leadtimeMin; i_leadtime<=leadtimeMax; i_leadtime++)
        {
            float maxVal = 0;

            // Get the maximum value of every station
            for (int i_station=0; i_station<stationsNb; i_station++)
            {
                float thisVal = 0;

                // Get values for return period
                float factor = 1;
                if(forecast->HasReferenceValues())
                {
                    float precip = forecast->GetReferenceValue(i_station, indexReferenceAxis);
                    wxASSERT(precip>0);
                    wxASSERT(precip<500);
                    factor = 1.0/precip;
                    wxASSERT(factor>0);
                }

                // Get values
                Array1DFloat theseVals = forecast->GetAnalogsValuesGross(i_leadtime, i_station);

                // Process percentiles
                if(asTools::HasNaN(&theseVals[0], &theseVals[theseVals.size()-1]))
                {
                    thisVal = NaNFloat;
                }
                else
                {
                    float forecastVal = asTools::Percentile(&theseVals[0], &theseVals[theseVals.size()-1], percentileThreshold);
                    wxASSERT_MSG(forecastVal>=0, wxString::Format("Forecast value = %g", forecastVal));
                    forecastVal *= factor;
                    thisVal = forecastVal;
                }

                // Keep it if higher
                if (thisVal>maxVal)
                {
                    maxVal = thisVal;
                }
            }

            allValues(i_model,i_leadtime) = maxVal;
        }
    }

    // Extract the highest values
    Array1DFloat values = allValues.colwise().maxCoeff();

    return values;
}

VectorString asResultsAnalogsForecastAggregator::ExtractMethodIds(std::vector <asResultsAnalogsForecast*> forecasts)
{
    VectorString methodsIds;

    methodsIds.push_back(forecasts[0]->GetMethodId());

    for (int i=1; i<forecasts.size(); i++)
    {
        wxString methodId = forecasts[i]->GetMethodId();
        bool addToArray = true;

        for (int j=1; j<methodsIds.size(); j++)
        {
            if (methodsIds[j].IsSameAs(methodId, false)) addToArray = false;
        }

        if (addToArray)
        {
            methodsIds.push_back(forecasts[i]->GetMethodId());
        }
    }

    return methodsIds;
}
