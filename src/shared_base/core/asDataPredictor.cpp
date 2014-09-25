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
 * Portions Copyright 2008-2013 University of Lausanne.
 */

#include "asDataPredictor.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>


asDataPredictor::asDataPredictor(const wxString &dataId)
{
    m_DataId = dataId;
    m_Level = 0;
    m_IsPreprocessed = false;
    m_CanBeClipped = true;
    m_LatPtsnb = 0;
    m_LonPtsnb = 0;
    m_LatIndexStep = 0;
    m_LonIndexStep = 0;
    m_PreprocessMethod = wxEmptyString;
    m_Initialized = false;
    m_AxesChecked = false;
    m_TimeZoneHours = 0.0;
    m_TimeStepHours = 0.0;
    m_FirstTimeStepHours = 0.0;
    m_UaxisStep = 0.0f;
    m_VaxisStep = 0.0f;
    m_UaxisShift = 0.0f;
    m_VaxisShift = 0.0f;
}

asDataPredictor::~asDataPredictor()
{

}

bool asDataPredictor::SetData(VArray2DFloat &val)
{
    wxASSERT(m_Time.size()>0);
    wxASSERT((int)m_Time.size()==(int)val.size());

    m_LatPtsnb = val[0].rows();
    m_LonPtsnb = val[0].cols();
    m_Data.clear();
    m_Data = val;

    return true;
}

bool asDataPredictor::Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray)
{
    if (!m_Initialized)
    {
        if (!Init()) {
            asLogError(wxString::Format(_("Error at initialization of the predictor dataset %s."), m_DatasetName.c_str()));
            return false;
        }
    }

    try
    {
        // Check the time array
        if(!CheckTimeArray(timeArray))
        {
            asLogError(_("The time array is not valid to load data."));
            return false;
        }

        // Create a new area matching the dataset
        asGeoAreaCompositeGrid* dataArea = CreateMatchingArea(desiredArea);

        // Store time array
        m_Time = timeArray.GetTimeArray();
        m_TimeIndexStep = wxMax(timeArray.GetTimeStepHours()/m_TimeStepHours, 1);
        
        // The desired level
        if (desiredArea)
        {
            m_Level = desiredArea->GetComposite(0).GetLevel();
        }

        // Number of composites
        int compositesNb = 1;
        if (dataArea)
        {
            compositesNb = dataArea->GetNbComposites();
        }

        // Extract composite data from files
        VVArray2DFloat compositeData(compositesNb);
        if (!ExtractFromFiles(dataArea, timeArray, compositeData))
        {
            asLogError(_("Extracting data from files failed."));
            wxDELETE(dataArea);
            return false;
        }
        
        // Merge the composites into m_Data
        if (!MergeComposites(compositeData, dataArea))
        {
            asLogError(_("Merging the composites failed."));
            wxDELETE(dataArea);
            return false;
        }

        // Interpolate the loaded data on the desired grid
        if (desiredArea && !InterpolateOnGrid(dataArea, desiredArea))
        {
            asLogError(_("Interpolation failed."));
            wxDELETE(dataArea);
            return false;
        }
        wxASSERT_MSG(m_Data[0].cols()==desiredArea->GetUaxisPtsnb(), wxString::Format("m_Data[0].cols()=%d, desiredArea->GetUaxisPtsnb()=%d", (int)m_Data[0].cols(), (int)desiredArea->GetUaxisPtsnb()));
        wxASSERT_MSG(m_Data[0].rows()==desiredArea->GetVaxisPtsnb(), wxString::Format("m_Data[0].rows()=%d, desiredArea->GetVaxisPtsnb()=%d", (int)m_Data[0].rows(), (int)desiredArea->GetVaxisPtsnb()));

        // Check the data container length
        if ((unsigned)m_Time.size()!=m_Data.size())
        {
            asLogError(_("The date and the data array lengths do not match."));
            wxDELETE(dataArea);
            return false;
        }

        wxDELETE(dataArea);
    }
    catch(bad_alloc& ba)
    {
        wxString msg(ba.what(), wxConvUTF8);
        asLogError(wxString::Format(_("Bad allocation caught when loading data: %s"), msg.c_str()));
        return false;
    }
    catch(asException& e)
    {
        wxString fullMessage = e.GetFullMessage();
        if (!fullMessage.IsEmpty())
        {
            asLogError(fullMessage);
        }
        asLogError(_("Failed to load data."));
        return false;
    }

    return true;
}

bool asDataPredictor::Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray)
{
    return Load(&desiredArea, timeArray);
}

bool asDataPredictor::Load(asGeoAreaCompositeGrid &desiredArea, double date)
{
    asTimeArray timeArray(date, asTimeArray::SingleDay);
    timeArray.Init();

    return Load(&desiredArea, timeArray);
}

bool asDataPredictor::Load(asGeoAreaCompositeGrid *desiredArea, double date)
{
    asTimeArray timeArray(date, asTimeArray::SingleDay);
    timeArray.Init();

    return Load(desiredArea, timeArray);
}

bool asDataPredictor::LoadFullArea(double date, float level)
{
    asTimeArray timeArray(date, asTimeArray::SingleDay);
    timeArray.Init();
    m_Level = level;

    return Load(NULL, timeArray);
}

asGeoAreaCompositeGrid* asDataPredictor::CreateMatchingArea(asGeoAreaCompositeGrid *desiredArea)
{
    if (desiredArea)
    {
        double dataUmin, dataVmin, dataUmax, dataVmax, dataUstep, dataVstep;
        int dataUptsnb, dataVptsnb;
        wxString gridType = desiredArea->GetGridTypeString();
        if (gridType.IsSameAs("Regular", false))
        {
            dataUmin = floor((desiredArea->GetAbsoluteUmin()-m_UaxisShift)/m_UaxisStep)*m_UaxisStep+m_UaxisShift;
            dataVmin = floor((desiredArea->GetAbsoluteVmin()-m_VaxisShift)/m_VaxisStep)*m_VaxisStep+m_VaxisShift;
            dataUmax = ceil((desiredArea->GetAbsoluteUmax()-m_UaxisShift)/m_UaxisStep)*m_UaxisStep+m_UaxisShift;
            dataVmax = ceil((desiredArea->GetAbsoluteVmax()-m_VaxisShift)/m_VaxisStep)*m_VaxisStep+m_VaxisShift;
            dataUstep = floor(desiredArea->GetUstep()/m_UaxisStep)*m_UaxisStep; // strides if allowed
            dataVstep = floor(desiredArea->GetVstep()/m_VaxisStep)*m_VaxisStep; // strides if allowed
            dataUptsnb = (dataUmax-dataUmin)/dataUstep+1;
            dataVptsnb = (dataVmax-dataVmin)/dataVstep+1;
        }
        else
        {
            dataUmin = desiredArea->GetAbsoluteUmin();
            dataVmin = desiredArea->GetAbsoluteVmin();
            dataUstep = desiredArea->GetUstep();
            dataVstep = desiredArea->GetVstep();
            dataUptsnb = desiredArea->GetUaxisPtsnb();
            dataVptsnb = desiredArea->GetVaxisPtsnb();
        }

        asGeoAreaCompositeGrid* dataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, dataUmin, dataUptsnb, dataUstep, dataVmin, dataVptsnb, dataVstep, desiredArea->GetLevel(), asNONE, asFLAT_ALLOWED);

        // Get indexes steps
        if (gridType.IsSameAs("Regular", false))
        {
            m_LonIndexStep = dataArea->GetUstep()/m_UaxisStep;
            m_LatIndexStep = dataArea->GetVstep()/m_VaxisStep;
        }
        else
        {
            m_LonIndexStep = 1;
            m_LatIndexStep = 1;
        }

        // Get axes length for preallocation
        m_LonPtsnb = dataArea->GetUaxisPtsnb();
        m_LatPtsnb = dataArea->GetVaxisPtsnb();

        return dataArea;
    }

    return NULL;
}

bool asDataPredictor::AdjustAxes(asGeoAreaCompositeGrid *dataArea, Array1DFloat &axisDataLon, Array1DFloat &axisDataLat, VVArray2DFloat &compositeData)
{
    if (!m_AxesChecked)
    {
        if (dataArea==NULL)
        {
            // Get axes length for preallocation
            m_LonPtsnb = axisDataLon.size();
            m_LatPtsnb = axisDataLat.size();
            m_AxisLon.resize(axisDataLon.size());
            m_AxisLon = axisDataLon;
            m_AxisLat.resize(axisDataLat.size());
            m_AxisLat = axisDataLat;
        }
        else
        {
            // Check that requested data do not overtake the file
            for (int i_comp=0; i_comp<dataArea->GetNbComposites(); i_comp++)
            {
                Array1DDouble axisLonComp = dataArea->GetUaxisComposite(i_comp);

                if (axisDataLon[axisDataLon.size()-1]>axisDataLon[0])
                {
                    wxASSERT(axisLonComp[axisLonComp.size()-1]>=axisLonComp[0]);

                    // Condition for change: The composite must not be fully outside (considered as handled) and the limit is not the coordinate grid border.
                    if (axisLonComp[axisLonComp.size()-1]>axisDataLon[axisDataLon.size()-1] && axisLonComp[0]<axisDataLon[axisDataLon.size()-1] && axisLonComp[axisLonComp.size()-1]!=dataArea->GetAxisUmax())
                    {
                        asLogMessage(_("Correcting the longitude extent according to the file limits."));
                        double Uwidth = axisDataLon[axisDataLon.size()-1]-dataArea->GetAbsoluteUmin();
                        wxASSERT(Uwidth>=0);
                        int Uptsnb = 1+Uwidth/dataArea->GetUstep();
                        asLogMessage(wxString::Format(_("Uptsnb = %d."), Uptsnb));
                        asGeoAreaCompositeGrid* newdataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, dataArea->GetGridTypeString(),
                                                                            dataArea->GetAbsoluteUmin(), Uptsnb,
                                                                            dataArea->GetUstep(), dataArea->GetAbsoluteVmin(),
                                                                            dataArea->GetVaxisPtsnb(), dataArea->GetVstep(),
                                                                            dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }
                }
                else
                {
                    wxASSERT(axisLonComp[axisLonComp.size()-1]>=axisLonComp[0]);

                    // Condition for change: The composite must not be fully outside (considered as handled) and the limit is not the coordinate grid border.
                    if (axisLonComp[axisLonComp.size()-1]>axisDataLon[0] && axisLonComp[0]<axisDataLon[0] && axisLonComp[axisLonComp.size()-1]!=dataArea->GetAxisUmax())
                    {
                        asLogMessage(_("Correcting the longitude extent according to the file limits."));
                        double Uwidth = axisDataLon[0]-dataArea->GetAbsoluteUmin();
                        wxASSERT(Uwidth>=0);
                        int Uptsnb = 1+Uwidth/dataArea->GetUstep();
                        asLogMessage(wxString::Format(_("Uptsnb = %d."), Uptsnb));
                        asGeoAreaCompositeGrid* newdataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, dataArea->GetGridTypeString(),
                                                                            dataArea->GetAbsoluteUmin(), Uptsnb,
                                                                            dataArea->GetUstep(), dataArea->GetAbsoluteVmin(),
                                                                            dataArea->GetVaxisPtsnb(), dataArea->GetVstep(),
                                                                            dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }
                }
            }

            Array1DDouble axisLon = dataArea->GetUaxis();
            m_AxisLon.resize(axisLon.size());
            for (int i=0; i<axisLon.size(); i++)
            {
                m_AxisLon[i] = (float)axisLon[i];
            }
            m_LonPtsnb = dataArea->GetUaxisPtsnb();
            wxASSERT_MSG(m_AxisLon.size()==m_LonPtsnb, wxString::Format("m_AxisLon.size()=%d, m_LonPtsnb=%d",(int)m_AxisLon.size(),m_LonPtsnb));

            // Check that requested data do not overtake the file
            for (int i_comp=0; i_comp<dataArea->GetNbComposites(); i_comp++)
            {
                Array1DDouble axisLatComp = dataArea->GetVaxisComposite(i_comp);

                if (axisDataLat[axisDataLat.size()-1]>axisDataLat[0])
                {
                    wxASSERT(axisLatComp[axisLatComp.size()-1]>=axisLatComp[0]);

                    // Condition for change: The composite must not be fully outside (considered as handled).
                    if (axisLatComp[axisLatComp.size()-1]>axisDataLat[axisDataLat.size()-1] && axisLatComp[0]<axisDataLat[axisDataLat.size()-1])
                    {
                        asLogMessage(_("Correcting the latitude extent according to the file limits."));
                        double Vwidth = axisDataLat[axisDataLat.size()-1]-dataArea->GetAbsoluteVmin();
                        wxASSERT(Vwidth>=0);
                        int Vptsnb = 1+Vwidth/dataArea->GetVstep();
                        asLogMessage(wxString::Format(_("Vptsnb = %d."), Vptsnb));
                        asGeoAreaCompositeGrid* newdataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, dataArea->GetGridTypeString(),
                                                                            dataArea->GetAbsoluteUmin(), dataArea->GetUaxisPtsnb(),
                                                                            dataArea->GetUstep(), dataArea->GetAbsoluteVmin(),
                                                                            Vptsnb, dataArea->GetVstep(),
                                                                            dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }

                }
                else
                {
                    wxASSERT(axisLatComp[axisLatComp.size()-1]>=axisLatComp[0]);

                    // Condition for change: The composite must not be fully outside (considered as handled).
                    if (axisLatComp[axisLatComp.size()-1]>axisDataLat[0] && axisLatComp[0]<axisDataLat[0])
                    {
                        asLogMessage(_("Correcting the latitude extent according to the file limits."));
                        double Vwidth = axisDataLat[0]-dataArea->GetAbsoluteVmin();
                        wxASSERT(Vwidth>=0);
                        int Vptsnb = 1+Vwidth/dataArea->GetVstep();
                        asLogMessage(wxString::Format(_("Vptsnb = %d."), Vptsnb));
                        asGeoAreaCompositeGrid* newdataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, dataArea->GetGridTypeString(),
                                                                            dataArea->GetAbsoluteUmin(), dataArea->GetUaxisPtsnb(),
                                                                            dataArea->GetUstep(), dataArea->GetAbsoluteVmin(),
                                                                            Vptsnb, dataArea->GetVstep(),
                                                                            dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }
                }
            }

            Array1DDouble axisLat = dataArea->GetVaxis();
            m_AxisLat.resize(axisLat.size());
            for (int i=0; i<axisLat.size(); i++)
            {
                // Latitude axis in reverse order
                m_AxisLat[i] = (float)axisLat[axisLat.size()-1-i];
            }
            m_LatPtsnb = dataArea->GetVaxisPtsnb();
            wxASSERT_MSG(m_AxisLat.size()==m_LatPtsnb, wxString::Format("m_AxisLat.size()=%d, m_LatPtsnb=%d",(int)m_AxisLat.size(),m_LatPtsnb));
        }

        compositeData = VVArray2DFloat(dataArea->GetNbComposites());
        m_AxesChecked = true;
    }

    return true;
}

bool asDataPredictor::Inline()
{
    //Already inlined
    if (m_LonPtsnb==1 || m_LatPtsnb==1)
    {
        return true;
    }

    wxASSERT(m_Data.size()>0);

    int timeSize = m_Data.size();
    int cols = m_Data[0].cols();
    int rows = m_Data[0].rows();

    Array2DFloat inlineData = Array2DFloat::Zero(1,cols*rows);

    VArray2DFloat newData;
    newData.reserve(m_Time.size()*m_LonPtsnb*m_LatPtsnb);
    newData.resize(timeSize);

    for (int i_time=0; i_time<timeSize; i_time++)
    {
        for (int i_row=0; i_row<rows; i_row++)
        {
            inlineData.block(0,i_row*cols,1,cols) = m_Data[i_time].row(i_row);
        }
        newData[i_time] = inlineData;
    }

    m_Data = newData;

    m_LatPtsnb = m_Data[0].rows();
    m_LonPtsnb = m_Data[0].cols();
    Array1DFloat emptyAxis(1);
    emptyAxis[0] = NaNFloat;
    m_AxisLat = emptyAxis;
    m_AxisLon = emptyAxis;

    return true;
}

bool asDataPredictor::MergeComposites(VVArray2DFloat &compositeData, asGeoAreaCompositeGrid *area)
{
    if (area)
    {
        // Get a container with the final size
        int sizeTime = compositeData[0].size();
        m_Data = VArray2DFloat(sizeTime, Array2DFloat(m_LatPtsnb,m_LonPtsnb));

        Array2DFloat blockUL, blockLL, blockUR, blockLR;
        int isblockUL = asNONE, isblockLL = asNONE, isblockUR = asNONE, isblockLR = asNONE;

        // Resize containers for composite areas
        for (int i_area = 0; i_area<area->GetNbComposites(); i_area++)
        {
            if((area->GetComposite(i_area).GetUmax()==area->GetUmax()) & (area->GetComposite(i_area).GetVmin()==area->GetVmin()))
            {
                blockUL.resize(compositeData[i_area][0].rows(),compositeData[i_area][0].cols());
                isblockUL = i_area;
            }
            else if((area->GetComposite(i_area).GetUmin()==area->GetUmin()) & (area->GetComposite(i_area).GetVmin()==area->GetVmin()))
            {
                blockUR.resize(compositeData[i_area][0].rows(),compositeData[i_area][0].cols());
                isblockUR = i_area;
            }
            else if((area->GetComposite(i_area).GetUmax()==area->GetUmax()) & (area->GetComposite(i_area).GetVmax()==area->GetVmax()))
            {
                blockLL.resize(compositeData[i_area][0].rows(),compositeData[i_area][0].cols());
                isblockLL = i_area;
            }
            else if((area->GetComposite(i_area).GetUmin()==area->GetUmin()) & (area->GetComposite(i_area).GetVmax()==area->GetVmax()))
            {
                blockLR.resize(compositeData[i_area][0].rows(),compositeData[i_area][0].cols());
                isblockLR = i_area;
            }
            else
            {
                asLogError(_("The data composite was not identified."));
                return false;
            }
        }

        // Merge the composite data together
        for (int i_time=0; i_time<sizeTime; i_time++)
        {
            // Append the composite areas
            for (int i_area = 0; i_area<area->GetNbComposites(); i_area++)
            {
                if(i_area == isblockUL)
                {
                    blockUL = compositeData[i_area][i_time];
                    m_Data[i_time].topLeftCorner(blockUL.rows(), blockUL.cols()) = blockUL;
                }
                else if(i_area == isblockUR)
                {
                    blockUR = compositeData[i_area][i_time];
                    m_Data[i_time].block(0, m_LonPtsnb-blockUR.cols(), blockUR.rows(), blockUR.cols()) = blockUR;
                }
                else if(i_area == isblockLL)
                {
                    blockLL = compositeData[i_area][i_time];
    // TODO (phorton#1#): Implement me!
                    asLogError(_("Not yet implemented."));
                    return false;
                }
                else if(i_area == isblockLR)
                {
                    blockLR = compositeData[i_area][i_time];
    // TODO (phorton#1#): Implement me!
                    asLogError(_("Not yet implemented."));
                    return false;
                }
                else
                {
                    asLogError(_("The data composite cannot be build."));
                    return false;
                }
            }
        }
    }
    else
    {
        m_Data = compositeData[0];
    }

    return true;
}

bool asDataPredictor::InterpolateOnGrid(asGeoAreaCompositeGrid *dataArea, asGeoAreaCompositeGrid *desiredArea)
{
    wxASSERT(dataArea);
    wxASSERT(desiredArea);
    bool changeUstart=false, changeUsteps=false, changeVstart=false, changeVsteps=false;

    // Check beginning on longitudes
    if (dataArea->GetAbsoluteUmin()!=desiredArea->GetAbsoluteUmin())
    {
        changeUstart = true;
    }

    // Check beginning on latitudes
    if (dataArea->GetAbsoluteVmin()!=desiredArea->GetAbsoluteVmin())
    {
        changeVstart = true;
    }

    // Check the cells size on longitudes
    if (!dataArea->GridsOverlay(desiredArea))
    {
        changeUsteps = true;
        changeVsteps = true;
    }

    // Proceed to the interpolation
    if (changeUstart || changeVstart || changeUsteps || changeVsteps)
    {
        // Containers for results
        int finalLengthLon = desiredArea->GetUaxisPtsnb();
        int finalLengthLat = desiredArea->GetVaxisPtsnb();
        VArray2DFloat latlonTimeData(m_Data.size(), Array2DFloat(finalLengthLat,finalLengthLon));

        // Creation of the axes
        Array1DFloat axisDataLon;
        if(dataArea->GetUaxisPtsnb()>1)
        {
            axisDataLon = Array1DFloat::LinSpaced(Eigen::Sequential, dataArea->GetUaxisPtsnb(), dataArea->GetAbsoluteUmin(), dataArea->GetAbsoluteUmax());
        }
        else
        {
            axisDataLon.resize(1);
            axisDataLon << dataArea->GetAbsoluteUmin();
        }

        Array1DFloat axisDataLat;
        if(dataArea->GetVaxisPtsnb()>1)
        {
            axisDataLat = Array1DFloat::LinSpaced(Eigen::Sequential, dataArea->GetVaxisPtsnb(), dataArea->GetAbsoluteVmax(), dataArea->GetAbsoluteVmin()); // From top to bottom
        }
        else
        {
            axisDataLat.resize(1);
            axisDataLat << dataArea->GetAbsoluteVmax();
        }

        Array1DFloat axisFinalLon;
        if(desiredArea->GetUaxisPtsnb()>1)
        {
            axisFinalLon = Array1DFloat::LinSpaced(Eigen::Sequential, desiredArea->GetUaxisPtsnb(), desiredArea->GetAbsoluteUmin(), desiredArea->GetAbsoluteUmax());
        }
        else
        {
            axisFinalLon.resize(1);
            axisFinalLon << desiredArea->GetAbsoluteUmin();
        }

        Array1DFloat axisFinalLat;
        if(desiredArea->GetVaxisPtsnb()>1)
        {
            axisFinalLat = Array1DFloat::LinSpaced(Eigen::Sequential, desiredArea->GetVaxisPtsnb(), desiredArea->GetAbsoluteVmax(), desiredArea->GetAbsoluteVmin()); // From top to bottom
        }
        else
        {
            axisFinalLat.resize(1);
            axisFinalLat << desiredArea->GetAbsoluteVmax();
        }

        // Indices
        int indexUfloor, indexUceil;
        int indexVfloor, indexVceil;
        int axisDataLonEnd = axisDataLon.size()-1;
        int axisDataLatEnd = axisDataLat.size()-1;

        // Pointer to last used element
        int indexLastLon=0, indexLastLat=0;

        // Variables
        double dU, dV;
        float valLLcorner, valULcorner, valLRcorner, valURcorner;

        // The interpolation loop
        for (unsigned int i_time=0; i_time<m_Data.size(); i_time++)
        {
            // Loop to extract the data from the array
            for (int i_lat=0; i_lat<finalLengthLat; i_lat++)
            {
                // Try the 2 next latitudes (from the top)
                if (axisDataLat.size()>indexLastLat+1 && axisDataLat[indexLastLat+1]==axisFinalLat[i_lat])
                {
                    indexVfloor = indexLastLat+1;
                    indexVceil = indexLastLat+1;
                }
                else if (axisDataLat.size()>indexLastLat+2 && axisDataLat[indexLastLat+2]==axisFinalLat[i_lat])
                {
                    indexVfloor = indexLastLat+2;
                    indexVceil = indexLastLat+2;
                }
                else
                {
                    // Search for floor and ceil
                    indexVfloor = indexLastLat + asTools::SortedArraySearchFloor(&axisDataLat[indexLastLat], &axisDataLat[axisDataLatEnd], axisFinalLat[i_lat]);
                    indexVceil = indexLastLat + asTools::SortedArraySearchCeil(&axisDataLat[indexLastLat], &axisDataLat[axisDataLatEnd], axisFinalLat[i_lat]);
                }

                if( indexVfloor==asOUT_OF_RANGE || indexVfloor==asNOT_FOUND || indexVceil==asOUT_OF_RANGE || indexVceil==asNOT_FOUND)
                {
                    asLogError(wxString::Format(_("The desired point is not available in the data for interpolation. Latitude %f was not found inbetween %f (index %d) to %f (index %d) (size = %d)."),
                                                axisFinalLat[i_lat], axisDataLat[indexLastLat], indexLastLat, axisDataLat[axisDataLatEnd], axisDataLatEnd, (int)axisDataLat.size()));
                    return false;
                }
                wxASSERT_MSG(indexVfloor>=0, wxString::Format("%f in %f to %f",axisFinalLat[i_lat], axisDataLat[indexLastLat], axisDataLat[axisDataLatEnd]));
                wxASSERT(indexVceil>=0);

                // Save last index
                indexLastLat = indexVfloor;

                for (int i_lon=0; i_lon<finalLengthLon; i_lon++)
                {
                    // Try the 2 next longitudes
                    if (axisDataLon.size()>indexLastLon+1 && axisDataLon[indexLastLon+1]==axisFinalLon[i_lon])
                    {
                        indexUfloor = indexLastLon+1;
                        indexUceil = indexLastLon+1;
                    }
                    else if (axisDataLon.size()>indexLastLon+2 && axisDataLon[indexLastLon+2]==axisFinalLon[i_lon])
                    {
                        indexUfloor = indexLastLon+2;
                        indexUceil = indexLastLon+2;
                    }
                    else
                    {
                        // Search for floor and ceil
                        indexUfloor = indexLastLon + asTools::SortedArraySearchFloor(&axisDataLon[indexLastLon], &axisDataLon[axisDataLonEnd], axisFinalLon[i_lon]);
                        indexUceil = indexLastLon + asTools::SortedArraySearchCeil(&axisDataLon[indexLastLon], &axisDataLon[axisDataLonEnd], axisFinalLon[i_lon]);
                    }

                    if( indexUfloor==asOUT_OF_RANGE || indexUfloor==asNOT_FOUND || indexUceil==asOUT_OF_RANGE || indexUceil==asNOT_FOUND)
                    {
                        asLogError(wxString::Format(_("The desired point is not available in the data for interpolation. Longitude %f was not found inbetween %f to %f."), axisFinalLon[i_lon], axisDataLon[indexLastLon], axisDataLon[axisDataLonEnd]));
                        return false;
                    }

                    wxASSERT(indexUfloor>=0);
                    wxASSERT(indexUceil>=0);

                    // Save last index
                    indexLastLon = indexUfloor;

                    // Proceed to the interpolation
                    if (indexUceil==indexUfloor)
                    {
                        dU = 0;
                    }
                    else
                    {
                        dU = (axisFinalLon[i_lon]-axisDataLon[indexUfloor])/(axisDataLon[indexUceil]-axisDataLon[indexUfloor]);
                    }
                    if (indexVceil==indexVfloor)
                    {
                        dV = 0;
                    }
                    else
                    {
                        dV = (axisFinalLat[i_lat]-axisDataLat[indexVfloor])/(axisDataLat[indexVceil]-axisDataLat[indexVfloor]);
                    }


                    if (dU==0 && dV==0)
                    {
                        latlonTimeData[i_time](i_lat, i_lon) = m_Data[i_time](indexVfloor, indexUfloor);
                    }
                    else if (dU==0)
                    {
                        valLLcorner = m_Data[i_time](indexVfloor, indexUfloor);
                        valULcorner = m_Data[i_time](indexVceil, indexUfloor);

                        latlonTimeData[i_time](i_lat, i_lon) =  (1-dU)*(1-dV)*valLLcorner
                                                  + (1-dU)*(dV)*valULcorner;
                    }
                    else if (dV==0)
                    {
                        valLLcorner = m_Data[i_time](indexVfloor, indexUfloor);
                        valLRcorner = m_Data[i_time](indexVfloor, indexUceil);

                        latlonTimeData[i_time](i_lat, i_lon) =  (1-dU)*(1-dV)*valLLcorner
                                                  + (dU)*(1-dV)*valLRcorner;
                    }
                    else
                    {
                        valLLcorner = m_Data[i_time](indexVfloor, indexUfloor);
                        valULcorner = m_Data[i_time](indexVceil, indexUfloor);
                        valLRcorner = m_Data[i_time](indexVfloor, indexUceil);
                        valURcorner = m_Data[i_time](indexVceil, indexUceil);

                        latlonTimeData[i_time](i_lat, i_lon) =  (1-dU)*(1-dV)*valLLcorner
                                                  + (1-dU)*(dV)*valULcorner
                                                  + (dU)*(1-dV)*valLRcorner
                                                  + (dU)*(dV)*valURcorner;
                    }
                }

                indexLastLon = 0;
            }

            indexLastLat = 0;
        }

        m_Data = latlonTimeData;
        m_LatPtsnb = finalLengthLat;
        m_LonPtsnb = finalLengthLon;
    }

    return true;
}
