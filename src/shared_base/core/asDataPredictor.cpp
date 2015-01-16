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
    m_XaxisStep = 0.0f;
    m_YaxisStep = 0.0f;
    m_XaxisShift = 0.0f;
    m_YaxisShift = 0.0f;
    m_FileAxisLatName = wxEmptyString;
    m_FileAxisLonName = wxEmptyString;
    m_FileAxisTimeName = wxEmptyString;
    m_FileAxisLevelName = wxEmptyString;
    m_FileExtension = wxEmptyString;
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
            wxASSERT(compositesNb>0);
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
        double dataXmin, dataYmin, dataXmax, dataYmax, dataXstep, dataYstep;
        int dataXptsnb, dataYptsnb;
        wxString gridType = desiredArea->GetGridTypeString();
        if (gridType.IsSameAs("Regular", false))
        {
            dataXmin = floor((desiredArea->GetAbsoluteXmin()-m_XaxisShift)/m_XaxisStep)*m_XaxisStep+m_XaxisShift;
            dataYmin = floor((desiredArea->GetAbsoluteYmin()-m_YaxisShift)/m_YaxisStep)*m_YaxisStep+m_YaxisShift;
            dataXmax = ceil((desiredArea->GetAbsoluteXmax()-m_XaxisShift)/m_XaxisStep)*m_XaxisStep+m_XaxisShift;
            dataYmax = ceil((desiredArea->GetAbsoluteYmax()-m_YaxisShift)/m_YaxisStep)*m_YaxisStep+m_YaxisShift;
            dataXstep = m_XaxisStep;
            dataYstep = m_YaxisStep;
            dataXptsnb = (dataXmax-dataXmin)/dataXstep+1;
            dataYptsnb = (dataYmax-dataYmin)/dataYstep+1;
        }
        else
        {
            dataXmin = desiredArea->GetAbsoluteXmin();
            dataYmin = desiredArea->GetAbsoluteYmin();
            dataXstep = desiredArea->GetXstep();
            dataYstep = desiredArea->GetYstep();
            dataXptsnb = desiredArea->GetXaxisPtsnb();
            dataYptsnb = desiredArea->GetYaxisPtsnb();
            if (!asTools::IsNaN(m_XaxisStep) && !asTools::IsNaN(m_YaxisStep) && (dataXstep!=m_XaxisStep || dataYstep!=m_YaxisStep))
            {
                asLogError(_("Interpolation is not allowed on irregular grids."));
                return NULL;
            }
        }

        asGeoAreaCompositeGrid* dataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, dataXmin, dataXptsnb, dataXstep, dataYmin, dataYptsnb, dataYstep, desiredArea->GetLevel(), asNONE, asFLAT_ALLOWED);

        // Get indexes steps
        if (gridType.IsSameAs("Regular", false))
        {
            m_LonIndexStep = dataArea->GetXstep()/m_XaxisStep;
            m_LatIndexStep = dataArea->GetYstep()/m_YaxisStep;
        }
        else
        {
            m_LonIndexStep = 1;
            m_LatIndexStep = 1;
        }

        // Get axes length for preallocation
        m_LonPtsnb = dataArea->GetXaxisPtsnb();
        m_LatPtsnb = dataArea->GetYaxisPtsnb();

        return dataArea;
    }

    return NULL;
}

asGeoAreaCompositeGrid* asDataPredictor::AdjustAxes(asGeoAreaCompositeGrid *dataArea, Array1DFloat &axisDataLon, Array1DFloat &axisDataLat, VVArray2DFloat &compositeData)
{
    if (!m_AxesChecked)
    {
        if (dataArea==NULL)
        {
            // Get axes length for preallocation
            m_LonPtsnb = axisDataLon.size();
            m_LatPtsnb = axisDataLat.size();
            m_AxisLon = axisDataLon;
            m_AxisLat = axisDataLat;
        }
        else
        {
            // Check that requested data do not overtake the file
            for (int i_comp=0; i_comp<dataArea->GetNbComposites(); i_comp++)
            {
                Array1DDouble axisLonComp = dataArea->GetXaxisComposite(i_comp);

                if (axisDataLon[axisDataLon.size()-1]>axisDataLon[0])
                {
                    wxASSERT(axisLonComp[axisLonComp.size()-1]>=axisLonComp[0]);

                    // Condition for change: The composite must not be fully outside (considered as handled) and the limit is not the coordinate grid border.
                    if (axisLonComp[axisLonComp.size()-1]>axisDataLon[axisDataLon.size()-1] && axisLonComp[0]<axisDataLon[axisDataLon.size()-1] && axisLonComp[axisLonComp.size()-1]!=dataArea->GetAxisXmax())
                    {
                        asLogMessage(_("Correcting the longitude extent according to the file limits."));
                        double Xwidth = axisDataLon[axisDataLon.size()-1]-dataArea->GetAbsoluteXmin();
                        wxASSERT(Xwidth>=0);
                        int Xptsnb = 1+Xwidth/dataArea->GetXstep();
                        asLogMessage(wxString::Format(_("Xptsnb = %d."), Xptsnb));
                        asGeoAreaCompositeGrid* newdataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, dataArea->GetGridTypeString(),
                                                                            dataArea->GetAbsoluteXmin(), Xptsnb,
                                                                            dataArea->GetXstep(), dataArea->GetAbsoluteYmin(),
                                                                            dataArea->GetYaxisPtsnb(), dataArea->GetYstep(),
                                                                            dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }
                }
                else
                {
                    wxASSERT(axisLonComp[axisLonComp.size()-1]>=axisLonComp[0]);

                    // Condition for change: The composite must not be fully outside (considered as handled) and the limit is not the coordinate grid border.
                    if (axisLonComp[axisLonComp.size()-1]>axisDataLon[0] && axisLonComp[0]<axisDataLon[0] && axisLonComp[axisLonComp.size()-1]!=dataArea->GetAxisXmax())
                    {
                        asLogMessage(_("Correcting the longitude extent according to the file limits."));
                        double Xwidth = axisDataLon[0]-dataArea->GetAbsoluteXmin();
                        wxASSERT(Xwidth>=0);
                        int Xptsnb = 1+Xwidth/dataArea->GetXstep();
                        asLogMessage(wxString::Format(_("Xptsnb = %d."), Xptsnb));
                        asGeoAreaCompositeGrid* newdataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, dataArea->GetGridTypeString(),
                                                                            dataArea->GetAbsoluteXmin(), Xptsnb,
                                                                            dataArea->GetXstep(), dataArea->GetAbsoluteYmin(),
                                                                            dataArea->GetYaxisPtsnb(), dataArea->GetYstep(),
                                                                            dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }
                }
            }

            Array1DDouble axisLon = dataArea->GetXaxis();
            m_AxisLon.resize(axisLon.size());
            for (int i=0; i<axisLon.size(); i++)
            {
                m_AxisLon[i] = (float)axisLon[i];
            }
            m_LonPtsnb = dataArea->GetXaxisPtsnb();
            wxASSERT_MSG(m_AxisLon.size()==m_LonPtsnb, wxString::Format("m_AxisLon.size()=%d, m_LonPtsnb=%d",(int)m_AxisLon.size(),m_LonPtsnb));

            // Check that requested data do not overtake the file
            for (int i_comp=0; i_comp<dataArea->GetNbComposites(); i_comp++)
            {
                Array1DDouble axisLatComp = dataArea->GetYaxisComposite(i_comp);

                if (axisDataLat[axisDataLat.size()-1]>axisDataLat[0])
                {
                    wxASSERT(axisLatComp[axisLatComp.size()-1]>=axisLatComp[0]);

                    // Condition for change: The composite must not be fully outside (considered as handled).
                    if (axisLatComp[axisLatComp.size()-1]>axisDataLat[axisDataLat.size()-1] && axisLatComp[0]<axisDataLat[axisDataLat.size()-1])
                    {
                        asLogMessage(_("Correcting the latitude extent according to the file limits."));
                        double Ywidth = axisDataLat[axisDataLat.size()-1]-dataArea->GetAbsoluteYmin();
                        wxASSERT(Ywidth>=0);
                        int Yptsnb = 1+Ywidth/dataArea->GetYstep();
                        asLogMessage(wxString::Format(_("Yptsnb = %d."), Yptsnb));
                        asGeoAreaCompositeGrid* newdataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, dataArea->GetGridTypeString(),
                                                                            dataArea->GetAbsoluteXmin(), dataArea->GetXaxisPtsnb(),
                                                                            dataArea->GetXstep(), dataArea->GetAbsoluteYmin(),
                                                                            Yptsnb, dataArea->GetYstep(),
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
                        double Ywidth = axisDataLat[0]-dataArea->GetAbsoluteYmin();
                        wxASSERT(Ywidth>=0);
                        int Yptsnb = 1+Ywidth/dataArea->GetYstep();
                        asLogMessage(wxString::Format(_("Yptsnb = %d."), Yptsnb));
                        asGeoAreaCompositeGrid* newdataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, dataArea->GetGridTypeString(),
                                                                            dataArea->GetAbsoluteXmin(), dataArea->GetXaxisPtsnb(),
                                                                            dataArea->GetXstep(), dataArea->GetAbsoluteYmin(),
                                                                            Yptsnb, dataArea->GetYstep(),
                                                                            dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }
                }
            }

            Array1DDouble axisLat = dataArea->GetYaxis();
            m_AxisLat.resize(axisLat.size());
            for (int i=0; i<axisLat.size(); i++)
            {
                // Latitude axis in reverse order
                m_AxisLat[i] = (float)axisLat[axisLat.size()-1-i];
            }
            m_LatPtsnb = dataArea->GetYaxisPtsnb();
            wxASSERT_MSG(m_AxisLat.size()==m_LatPtsnb, wxString::Format("m_AxisLat.size()=%d, m_LatPtsnb=%d",(int)m_AxisLat.size(),m_LatPtsnb));
        }

        compositeData = VVArray2DFloat(dataArea->GetNbComposites());
        m_AxesChecked = true;
    }

    return dataArea;
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
            if((area->GetComposite(i_area).GetXmax()==area->GetXmax()) & (area->GetComposite(i_area).GetYmin()==area->GetYmin()))
            {
                blockUL.resize(compositeData[i_area][0].rows(),compositeData[i_area][0].cols());
                isblockUL = i_area;
            }
            else if((area->GetComposite(i_area).GetXmin()==area->GetXmin()) & (area->GetComposite(i_area).GetYmin()==area->GetYmin()))
            {
                blockUR.resize(compositeData[i_area][0].rows(),compositeData[i_area][0].cols());
                isblockUR = i_area;
            }
            else if((area->GetComposite(i_area).GetXmax()==area->GetXmax()) & (area->GetComposite(i_area).GetYmax()==area->GetYmax()))
            {
                blockLL.resize(compositeData[i_area][0].rows(),compositeData[i_area][0].cols());
                isblockLL = i_area;
            }
            else if((area->GetComposite(i_area).GetXmin()==area->GetXmin()) & (area->GetComposite(i_area).GetYmax()==area->GetYmax()))
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
    wxASSERT(dataArea->GetNbComposites()>0);
    wxASSERT(desiredArea);
    wxASSERT(desiredArea->GetNbComposites()>0);
    bool changeXstart=false, changeXsteps=false, changeYstart=false, changeYsteps=false;

    // Check beginning on longitudes
    if (dataArea->GetAbsoluteXmin()!=desiredArea->GetAbsoluteXmin())
    {
        changeXstart = true;
    }

    // Check beginning on latitudes
    if (dataArea->GetAbsoluteYmin()!=desiredArea->GetAbsoluteYmin())
    {
        changeYstart = true;
    }

    // Check the cells size on longitudes
    if (!dataArea->GridsOverlay(desiredArea))
    {
        changeXsteps = true;
        changeYsteps = true;
    }

    // Proceed to the interpolation
    if (changeXstart || changeYstart || changeXsteps || changeYsteps)
    {
        // Containers for results
        int finalLengthLon = desiredArea->GetXaxisPtsnb();
        int finalLengthLat = desiredArea->GetYaxisPtsnb();
        VArray2DFloat latlonTimeData(m_Data.size(), Array2DFloat(finalLengthLat,finalLengthLon));

        // Creation of the axes
        Array1DFloat axisDataLon;
        if(dataArea->GetXaxisPtsnb()>1)
        {
            axisDataLon = Array1DFloat::LinSpaced(Eigen::Sequential, dataArea->GetXaxisPtsnb(), dataArea->GetAbsoluteXmin(), dataArea->GetAbsoluteXmax());
        }
        else
        {
            axisDataLon.resize(1);
            axisDataLon << dataArea->GetAbsoluteXmin();
        }

        Array1DFloat axisDataLat;
        if(dataArea->GetYaxisPtsnb()>1)
        {
            axisDataLat = Array1DFloat::LinSpaced(Eigen::Sequential, dataArea->GetYaxisPtsnb(), dataArea->GetAbsoluteYmax(), dataArea->GetAbsoluteYmin()); // From top to bottom
        }
        else
        {
            axisDataLat.resize(1);
            axisDataLat << dataArea->GetAbsoluteYmax();
        }

        Array1DFloat axisFinalLon;
        if(desiredArea->GetXaxisPtsnb()>1)
        {
            axisFinalLon = Array1DFloat::LinSpaced(Eigen::Sequential, desiredArea->GetXaxisPtsnb(), desiredArea->GetAbsoluteXmin(), desiredArea->GetAbsoluteXmax());
        }
        else
        {
            axisFinalLon.resize(1);
            axisFinalLon << desiredArea->GetAbsoluteXmin();
        }

        Array1DFloat axisFinalLat;
        if(desiredArea->GetYaxisPtsnb()>1)
        {
            axisFinalLat = Array1DFloat::LinSpaced(Eigen::Sequential, desiredArea->GetYaxisPtsnb(), desiredArea->GetAbsoluteYmax(), desiredArea->GetAbsoluteYmin()); // From top to bottom
        }
        else
        {
            axisFinalLat.resize(1);
            axisFinalLat << desiredArea->GetAbsoluteYmax();
        }

        // Indices
        int indexXfloor, indexXceil;
        int indexYfloor, indexYceil;
        int axisDataLonEnd = axisDataLon.size()-1;
        int axisDataLatEnd = axisDataLat.size()-1;

        // Pointer to last used element
        int indexLastLon=0, indexLastLat=0;

        // Variables
        double dX, dY;
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
                    indexYfloor = indexLastLat+1;
                    indexYceil = indexLastLat+1;
                }
                else if (axisDataLat.size()>indexLastLat+2 && axisDataLat[indexLastLat+2]==axisFinalLat[i_lat])
                {
                    indexYfloor = indexLastLat+2;
                    indexYceil = indexLastLat+2;
                }
                else
                {
                    // Search for floor and ceil
                    indexYfloor = indexLastLat + asTools::SortedArraySearchFloor(&axisDataLat[indexLastLat], &axisDataLat[axisDataLatEnd], axisFinalLat[i_lat]);
                    indexYceil = indexLastLat + asTools::SortedArraySearchCeil(&axisDataLat[indexLastLat], &axisDataLat[axisDataLatEnd], axisFinalLat[i_lat]);
                }

                if( indexYfloor==asOUT_OF_RANGE || indexYfloor==asNOT_FOUND || indexYceil==asOUT_OF_RANGE || indexYceil==asNOT_FOUND)
                {
                    asLogError(wxString::Format(_("The desired point is not available in the data for interpolation. Latitude %f was not found inbetween %f (index %d) to %f (index %d) (size = %d)."),
                                                axisFinalLat[i_lat], axisDataLat[indexLastLat], indexLastLat, axisDataLat[axisDataLatEnd], axisDataLatEnd, (int)axisDataLat.size()));
                    return false;
                }
                wxASSERT_MSG(indexYfloor>=0, wxString::Format("%f in %f to %f",axisFinalLat[i_lat], axisDataLat[indexLastLat], axisDataLat[axisDataLatEnd]));
                wxASSERT(indexYceil>=0);

                // Save last index
                indexLastLat = indexYfloor;

                for (int i_lon=0; i_lon<finalLengthLon; i_lon++)
                {
                    // Try the 2 next longitudes
                    if (axisDataLon.size()>indexLastLon+1 && axisDataLon[indexLastLon+1]==axisFinalLon[i_lon])
                    {
                        indexXfloor = indexLastLon+1;
                        indexXceil = indexLastLon+1;
                    }
                    else if (axisDataLon.size()>indexLastLon+2 && axisDataLon[indexLastLon+2]==axisFinalLon[i_lon])
                    {
                        indexXfloor = indexLastLon+2;
                        indexXceil = indexLastLon+2;
                    }
                    else
                    {
                        // Search for floor and ceil
                        indexXfloor = indexLastLon + asTools::SortedArraySearchFloor(&axisDataLon[indexLastLon], &axisDataLon[axisDataLonEnd], axisFinalLon[i_lon]);
                        indexXceil = indexLastLon + asTools::SortedArraySearchCeil(&axisDataLon[indexLastLon], &axisDataLon[axisDataLonEnd], axisFinalLon[i_lon]);
                    }

                    if( indexXfloor==asOUT_OF_RANGE || indexXfloor==asNOT_FOUND || indexXceil==asOUT_OF_RANGE || indexXceil==asNOT_FOUND)
                    {
                        asLogError(wxString::Format(_("The desired point is not available in the data for interpolation. Longitude %f was not found inbetween %f to %f."), axisFinalLon[i_lon], axisDataLon[indexLastLon], axisDataLon[axisDataLonEnd]));
                        return false;
                    }

                    wxASSERT(indexXfloor>=0);
                    wxASSERT(indexXceil>=0);

                    // Save last index
                    indexLastLon = indexXfloor;

                    // Proceed to the interpolation
                    if (indexXceil==indexXfloor)
                    {
                        dX = 0;
                    }
                    else
                    {
                        dX = (axisFinalLon[i_lon]-axisDataLon[indexXfloor])/(axisDataLon[indexXceil]-axisDataLon[indexXfloor]);
                    }
                    if (indexYceil==indexYfloor)
                    {
                        dY = 0;
                    }
                    else
                    {
                        dY = (axisFinalLat[i_lat]-axisDataLat[indexYfloor])/(axisDataLat[indexYceil]-axisDataLat[indexYfloor]);
                    }


                    if (dX==0 && dY==0)
                    {
                        latlonTimeData[i_time](i_lat, i_lon) = m_Data[i_time](indexYfloor, indexXfloor);
                    }
                    else if (dX==0)
                    {
                        valLLcorner = m_Data[i_time](indexYfloor, indexXfloor);
                        valULcorner = m_Data[i_time](indexYceil, indexXfloor);

                        latlonTimeData[i_time](i_lat, i_lon) =  (1-dX)*(1-dY)*valLLcorner
                                                  + (1-dX)*(dY)*valULcorner;
                    }
                    else if (dY==0)
                    {
                        valLLcorner = m_Data[i_time](indexYfloor, indexXfloor);
                        valLRcorner = m_Data[i_time](indexYfloor, indexXceil);

                        latlonTimeData[i_time](i_lat, i_lon) =  (1-dX)*(1-dY)*valLLcorner
                                                  + (dX)*(1-dY)*valLRcorner;
                    }
                    else
                    {
                        valLLcorner = m_Data[i_time](indexYfloor, indexXfloor);
                        valULcorner = m_Data[i_time](indexYceil, indexXfloor);
                        valLRcorner = m_Data[i_time](indexYfloor, indexXceil);
                        valURcorner = m_Data[i_time](indexYceil, indexXceil);

                        latlonTimeData[i_time](i_lat, i_lon) =  (1-dX)*(1-dY)*valLLcorner
                                                  + (1-dX)*(dY)*valULcorner
                                                  + (dX)*(1-dY)*valLRcorner
                                                  + (dX)*(dY)*valURcorner;
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
