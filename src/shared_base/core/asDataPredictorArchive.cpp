/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#include "asDataPredictorArchive.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>


asDataPredictorArchive::asDataPredictorArchive(asCatalogPredictorsArchive &catalog)
:
asDataPredictor()
{
    m_Catalog = catalog;
}

asDataPredictorArchive::~asDataPredictorArchive()
{

}

bool asDataPredictorArchive::LoadFullArea(double date, float level, const wxString &AlternatePredictorDataPath)
{
    asTimeArray timeArray(date, asTimeArray::SingleDay);
    timeArray.Init();
    m_Level = level;

    return Load(NULL, timeArray, AlternatePredictorDataPath);
}

bool asDataPredictorArchive::Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray, const VectorString &AlternatePredictorDataPath)
{
    wxString AlternatePredictorDataPathStr = wxEmptyString;
    if (AlternatePredictorDataPath.size()>0)
    {
        AlternatePredictorDataPathStr = AlternatePredictorDataPath[0];
    }

    return Load(&desiredArea, timeArray, AlternatePredictorDataPathStr);
}

bool asDataPredictorArchive::Load(asGeoAreaCompositeGrid &desiredArea, double date, const wxString &AlternatePredictorDataPath)
{
    asTimeArray timeArray(date, asTimeArray::SingleDay);
    timeArray.Init();

    return Load(&desiredArea, timeArray, AlternatePredictorDataPath);
}

bool asDataPredictorArchive::Load(asGeoAreaCompositeGrid *desiredArea, double date, const wxString &AlternatePredictorDataPath)
{
    asTimeArray timeArray(date, asTimeArray::SingleDay);
    timeArray.Init();

    return Load(desiredArea, timeArray, AlternatePredictorDataPath);
}

bool asDataPredictorArchive::Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray, const wxString &AlternatePredictorDataPath)
{
    return Load(&desiredArea, timeArray, AlternatePredictorDataPath);
}

bool asDataPredictorArchive::Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray, const wxString &AlternatePredictorDataPath)
{
    try
    {
        // Check the time array
        if(!asDataPredictorArchive::CheckTimeArray(timeArray))
        {
            asLogError(_("The time array is not valid to load data."));
            return false;
        }

        // File path
        wxChar dateStartTag = '[', dateEndTag = ']';
        wxString fileName = m_Catalog.GetDataInfo().FileName;
        wxString filePath;
        if(AlternatePredictorDataPath.IsEmpty())
        {
            filePath = m_Catalog.GetDataPath();
        }
        else
        {
            filePath = AlternatePredictorDataPath;
        }
        if ( (filePath.Last()!='/') && (filePath.Last()!='\\') )
        {
            filePath.Append('/');
        }

        // Get the boundary years
        int yearFirst = 0, yearLast = 0;
        if (m_Catalog.GetDataFileLength()==Year)
        {
            yearFirst = timeArray.GetFirstDayYear();
            yearLast = timeArray.GetLastDayYear();
        }
        else if (m_Catalog.GetDataFileLength()==Total)
        {
            yearFirst = 0;
            yearLast = 0;
        }

        // The data name
        wxString dataName = m_Catalog.GetDataInfo().FileVarName;
        if (dataName.IsEmpty())
        {
            asLogError(_("The archive predictor variable name is not defined."));
            return false;
        }

        // Create a new area matching the dataset
        asGeoAreaCompositeGrid* dataArea = NULL;
        size_t indexStepLon = 1, indexStepLat = 1, indexStepTime = 1;
        if (desiredArea)
        {
            double dataUmin, dataVmin, dataUmax, dataVmax, dataUstep, dataVstep;
            int dataUptsnb, dataVptsnb;
            wxString gridType = desiredArea->GetGridTypeString();
            if (gridType.IsSameAs("Regular", false))
            {
                double dataUstepCat = m_Catalog.GetDataUaxisStep();
                double dataVstepCat = m_Catalog.GetDataVaxisStep();
                double dataUshiftCat = m_Catalog.GetDataUaxisShift();
                double dataVshiftCat = m_Catalog.GetDataVaxisShift();
                dataUmin = floor((desiredArea->GetAbsoluteUmin()-dataUshiftCat)/dataUstepCat)*dataUstepCat+dataUshiftCat;
                dataVmin = floor((desiredArea->GetAbsoluteVmin()-dataVshiftCat)/dataVstepCat)*dataVstepCat+dataVshiftCat;
                dataUmax = ceil((desiredArea->GetAbsoluteUmax()-dataUshiftCat)/dataUstepCat)*dataUstepCat+dataUshiftCat;
                dataVmax = ceil((desiredArea->GetAbsoluteVmax()-dataVshiftCat)/dataVstepCat)*dataVstepCat+dataVshiftCat;
                dataUstep = floor(desiredArea->GetUstep()/dataUstepCat)*dataUstepCat; // NetCDF allows to use strides
                dataVstep = floor(desiredArea->GetVstep()/dataVstepCat)*dataVstepCat; // NetCDF allows to use strides
                dataUptsnb = (dataUmax-dataUmin)/dataUstep+1;
                dataVptsnb = (dataVmax-dataVmin)/dataVstep+1;
            }
            else
            {
                dataUmin = desiredArea->GetAbsoluteUmin();
                dataVmin = desiredArea->GetAbsoluteVmin();
                dataUmax = desiredArea->GetAbsoluteUmax();
                dataVmax = desiredArea->GetAbsoluteVmax();
                dataUstep = desiredArea->GetUstep();
                dataVstep = desiredArea->GetVstep();
                dataUptsnb = desiredArea->GetUaxisPtsnb();
                dataVptsnb = desiredArea->GetVaxisPtsnb();
            }

            dataArea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, dataUmin, dataUptsnb, dataUstep, dataVmin, dataVptsnb, dataVstep, desiredArea->GetLevel(), asNONE, asFLAT_ALLOWED);

            // Get indexes steps
            if (gridType.IsSameAs("Regular", false))
            {
                double dataUstepCat = m_Catalog.GetDataUaxisStep();
                double dataVstepCat = m_Catalog.GetDataVaxisStep();
                indexStepLon = dataArea->GetUstep()/dataUstepCat;
                indexStepLat = dataArea->GetVstep()/dataVstepCat;
            }
            else
            {
                indexStepLon = 1;
                indexStepLat = 1;
            }

            // Get axes length for preallocation
            GetSizes(*dataArea, timeArray);
            InitContainers();
        }
        else
        {
            m_SizeTime = timeArray.GetSize();
            m_Time.resize(m_SizeTime);
        }

        indexStepTime = timeArray.GetTimeStepHours()/m_Catalog.GetTimeStepHours();
        indexStepTime = wxMax(indexStepTime,1);

        // Add dates to m_Time
        m_Time = timeArray.GetTimeArray();

        // Check of the array length
        int counterTime = 0;

        // The desired level
        if (desiredArea)
        {
            m_Level = desiredArea->GetComposite(0).GetLevel();
        }

        // Containers for the axes
        Array1DFloat axisDataLon, axisDataLat;

        #ifndef UNIT_TESTING
            #if wxUSE_GUI
                // The progress bar
                asDialogProgressBar progressBar(_("Loading data from files.\n"), yearLast-yearFirst);
            #endif
        #endif

        // Loop through the files
        for (int i_year=yearFirst; i_year<=yearLast; i_year++)
        {
            // Build the file path
            wxString fileFullPath;
            if (m_Catalog.GetDataFileLength()==Year)
            {
                fileFullPath = filePath + fileName.Before(dateStartTag) + wxString::Format("%d", (int)i_year) + fileName.After(dateEndTag);
            }
            else if (m_Catalog.GetDataFileLength()==Total)
            {
                fileFullPath = filePath + fileName;
            }

            #ifndef UNIT_TESTING
                #if wxUSE_GUI
                    // Update the progress bar
                    if (m_Catalog.GetDataFileLength()==Year)
                    {
                        wxString fileNameMessage = wxString::Format(_("Loading data from files.\nFile: %s%d%s"), fileName.Before(dateStartTag).c_str(), i_year, fileName.After(dateEndTag).c_str());
                        if(!progressBar.Update(i_year-yearFirst, fileNameMessage))
                        {
                            asLogWarning(_("The process has been canceled by the user."));
                            wxDELETE(dataArea);
                            return false;
                        }
                    }
                #endif
            #endif

            ThreadsManager().CritSectionNetCDF().Enter();

            // Open the NetCDF file
            asFileNetcdf ncFile(fileFullPath, asFileNetcdf::ReadOnly);
            if(!ncFile.Open())
            {
                wxDELETE(dataArea);
                ThreadsManager().CritSectionNetCDF().Leave();
                return false;
            }

            // Number of dimensions
            int nDims = ncFile.GetNDims();
            wxASSERT(nDims>=3);
            wxASSERT(nDims<=4);

            // Get some attributes
            float dataAddOffset = ncFile.GetAttFloat("add_offset", dataName);
            if (asTools::IsNaN(dataAddOffset)) dataAddOffset = 0;
            float dataScaleFactor = ncFile.GetAttFloat("scale_factor", dataName);
            if (asTools::IsNaN(dataScaleFactor)) dataScaleFactor = 1;
            bool scalingNeeded = true;
            if (dataAddOffset==0 && dataScaleFactor==1) scalingNeeded = false;

            // Get full axes from the netcdf file
                // Longitudes
            size_t axisDataLonLength = ncFile.GetVarLength("lon");
            axisDataLon.resize(axisDataLonLength);
            ncFile.GetVar("lon", &axisDataLon[0]);
                // Latitudes
            size_t axisDataLatLength = ncFile.GetVarLength("lat");
            axisDataLat.resize(axisDataLatLength);
            ncFile.GetVar("lat", &axisDataLat[0]);
                // Levels
            size_t axisDataLevelLength = 0;
            Array1DFloat axisDataLevel;
            if (nDims==4)
            {
                axisDataLevelLength = ncFile.GetVarLength("level");
                axisDataLevel.resize(axisDataLevelLength);
                ncFile.GetVar("level", &axisDataLevel[0]);
            }
                // Time
            size_t axisDataTimeLength = ncFile.GetVarLength("time");
            // Time array takes ages to load !! Avoid if possible. Get the first value of the time array.
            double valFirstTime = ncFile.GetVarOneDouble("time", 0);
            valFirstTime = (valFirstTime/24.0); // hours to days

            if (m_Catalog.GetSetId().IsSameAs("NCEP_R-1", false))
            {
                valFirstTime += asTime::GetMJD(1,1,1); // to MJD: add a negative time span
            }
            else if (m_Catalog.GetSetId().IsSameAs("NCEP_R-1_subset"))
            {
                valFirstTime += asTime::GetMJD(1,1,1); // to MJD: add a negative time span
            }
            else if (m_Catalog.GetSetId().IsSameAs("NCEP_R-2"))
            {
                valFirstTime += asTime::GetMJD(1800,1,1); // to MJD: add a negative time span
            }
            else if (m_Catalog.GetSetId().IsSameAs("NCEP_R-2_subset"))
            {
                valFirstTime += asTime::GetMJD(1800,1,1); // to MJD: add a negative time span
            }
            else if (m_Catalog.GetSetId().IsSameAs("NOAA_OISST-2_subset"))
            {
                valFirstTime += asTime::GetMJD(1,1,1); // to MJD: add a negative time span
            }
            else if (m_Catalog.GetSetId().IsSameAs("NOAA_OISST-2_subset_1deg"))
            {
                valFirstTime += asTime::GetMJD(1,1,1); // to MJD: add a negative time span
            }
            else
            {
                if (!g_UnitTesting)
                {
                    asLogError("The dataset time start is not correcty handled: dataset id not recognized");
                    return false;
                }
                else
                {
                    valFirstTime += asTime::GetMJD(1,1,1); // to MJD: add a negative time span
                }
            }

            if (desiredArea==NULL && i_year==yearFirst)
            {
                // Get axes length for preallocation
                m_LonPtsnb = axisDataLonLength;
                m_LatPtsnb = axisDataLatLength;
                m_AxisLon.resize(axisDataLon.size());
                m_AxisLon = axisDataLon;
                m_AxisLat.resize(axisDataLat.size());
                m_AxisLat = axisDataLat;
                m_Data.reserve(m_SizeTime*m_LonPtsnb*m_LatPtsnb);
            }
            else if(desiredArea!=NULL && i_year==yearFirst)
            {
                // Check that requested data do not overtake the file
                for (int i_comp=0; i_comp<dataArea->GetNbComposites(); i_comp++)
                {
                    Array1DDouble axisLonComp = dataArea->GetUaxisComposite(i_comp);
                    if (axisDataLon[axisDataLonLength-1]>axisDataLon[0])
                    {
                        wxASSERT(axisLonComp[axisLonComp.size()-1]>=axisLonComp[0]);
                        // Condition for change: The composite must not be fully outside (considered as handled) and the limit is not the coordinate grid border.
                        if (axisLonComp[axisLonComp.size()-1]>axisDataLon[axisDataLonLength-1] && axisLonComp[0]<axisDataLon[axisDataLonLength-1] && axisLonComp[axisLonComp.size()-1]!=dataArea->GetAxisUmax())
                        {
                            asLogMessage(_("Correcting the longitude extent according to the file limits."));
                            double Uwidth = axisDataLon[axisDataLonLength-1]-dataArea->GetAbsoluteUmin();
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
                    if (axisDataLat[axisDataLatLength-1]>axisDataLat[0])
                    {
                        wxASSERT(axisLatComp[axisLatComp.size()-1]>=axisLatComp[0]);
                        // Condition for change: The composite must not be fully outside (considered as handled).
                        if (axisLatComp[axisLatComp.size()-1]>axisDataLat[axisDataLatLength-1] && axisLatComp[0]<axisDataLat[axisDataLatLength-1])
                        {
                            asLogMessage(_("Correcting the latitude extent according to the file limits."));
                            double Vwidth = axisDataLat[axisDataLatLength-1]-dataArea->GetAbsoluteVmin();
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

                m_Data.reserve(m_SizeTime*m_LonPtsnb*m_LatPtsnb);
            }

            // Get start and end of the current file
            double timeStart = 0, timeEnd = 0;
            if (m_Catalog.GetDataFileLength()==Year)
            {
                // Get start and end of the current year
                timeStart = asTime::GetMJD(i_year,1,1,0,0);
                timeEnd = asTime::GetMJD(i_year,12,31,23,59);
            }
            else if (m_Catalog.GetDataFileLength()==Total)
            {
                // Get start and end of the serie
                timeStart = timeArray.GetFirst();
                timeEnd = timeArray.GetLast();
            }

            // Get the time length
            double timeArrayIndexStart = timeArray.GetIndexFirstAfter(timeStart);
            double timeArrayIndexEnd = timeArray.GetIndexFirstBefore(timeEnd);
            int indexLengthTime = timeArrayIndexEnd-timeArrayIndexStart+1;
            int indexLengthTimeArray = indexLengthTime;

            // Correct the time start and end
            size_t indexStartTime = 0;
            int cutStart = 0;
            if(m_Catalog.GetDataFileLength()==Total || i_year==yearFirst)
            {
                cutStart = timeArrayIndexStart;
            }
            int cutEnd = 0;
            while (valFirstTime<timeArray[timeArrayIndexStart])
            {
                valFirstTime += m_Catalog.GetTimeStepDays();
                indexStartTime++;
            }
            if (indexStartTime+indexLengthTime>axisDataTimeLength)
            {
                indexLengthTime--;
                cutEnd++;
            }

            // Get the number of iterations
            int iterationNb = 1;
            if (desiredArea)
            {
                iterationNb = dataArea->GetNbComposites();
            }

            // Containers for extraction
            VectorInt vectIndexLengthLat;
            VectorInt vectIndexLengthLon;
            VectorBool vectLoad360;
            VectorInt vectTotLength;
            VVectorShort vectData;
            VVectorShort vectData360;

            for (int i_area = 0; i_area<iterationNb; i_area++)
            {
                // Check if necessary to load the data of lon=360 (so lon=0)
                bool load360 = false;

                int indexStartLon, indexStartLat, indexLengthLon, indexLengthLat;
                if (desiredArea)
                {
                    // Get the spatial extent
                    float lonMin = dataArea->GetUaxisCompositeStart(i_area);
                    float lonMax = dataArea->GetUaxisCompositeEnd(i_area);
                    float latMinStart = dataArea->GetVaxisCompositeStart(i_area);
                    float latMinEnd = dataArea->GetVaxisCompositeEnd(i_area);

                    // The dimensions lengths
                    indexLengthLon = dataArea->GetUaxisCompositePtsnb(i_area);
                    indexLengthLat = dataArea->GetVaxisCompositePtsnb(i_area);

                    if(lonMax==dataArea->GetAxisUmax())
                    {
                        // Correction if the lon 360° is required (doesn't exist)
                        load360 = true;
                        for (int i_check = 0; i_check<iterationNb; i_check++)
                        {
                            // If so, already loaded in another composite
                            if(dataArea->GetComposite(i_check).GetUmin() == 0)
                            {
                                load360 = false;
                            }
                        }
                        lonMax -= dataArea->GetUstep();
                        indexLengthLon--;
                    }

                    // Get the spatial indices of the desired data
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], lonMin, 0.01f);
                    if(indexStartLon==asOUT_OF_RANGE)
                    {
                        // If not found, try with negative angles
                        indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], lonMin-360, 0.01f);
                    }
                    if(indexStartLon==asOUT_OF_RANGE)
                    {
                        // If not found, try with angles above 360°
                        indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], lonMin+360, 0.01f);
                    }
                    if(indexStartLon<0)
                    {
                        asLogError(wxString::Format("Cannot find lonMin (%f) in the array axisDataLon ([0]=%f -> [%d]=%f) ", lonMin, axisDataLon[0], (int)axisDataLonLength, axisDataLon[axisDataLonLength-1]));
                        return false;
                    }
                    wxASSERT_MSG(indexStartLon>=0, wxString::Format("axisDataLon[0] = %f, &axisDataLon[%d] = %f & lonMin = %f", axisDataLon[0], (int)axisDataLonLength, axisDataLon[axisDataLonLength-1], lonMin));

                    int indexStartLat1 = asTools::SortedArraySearch(&axisDataLat[0], &axisDataLat[axisDataLatLength-1], latMinStart, 0.01f);
                    int indexStartLat2 = asTools::SortedArraySearch(&axisDataLat[0], &axisDataLat[axisDataLatLength-1], latMinEnd, 0.01f);
                    wxASSERT_MSG(indexStartLat1>=0, wxString::Format("Looking for %g in %g to %g", latMinStart, axisDataLat[0], axisDataLat[axisDataLatLength-1]));
                    wxASSERT_MSG(indexStartLat2>=0, wxString::Format("Looking for %g in %g to %g", latMinEnd, axisDataLat[0], axisDataLat[axisDataLatLength-1]));
                    indexStartLat = wxMin(indexStartLat1, indexStartLat2);
                }
                else
                {
                    indexStartLon = 0;
                    indexStartLat = 0;
                    indexLengthLon = m_LonPtsnb;
                    indexLengthLat = m_LatPtsnb;
                }
                int indexLevel = 0;
                if (nDims==4)
                {
                    indexLevel = asTools::SortedArraySearch(&axisDataLevel[0], &axisDataLevel[axisDataLevelLength-1], m_Level, 0.01f);
                }

                // Create the arrays to receive the data
                VectorShort data, data360;

                // Resize the arrays to store the new data
                int totLength = indexLengthTimeArray * indexLengthLat * indexLengthLon;
                wxASSERT(totLength>0);
                data.resize(totLength);

                // Fill empty begining with NaNs
                int indexBegining = 0;
                if(cutStart>0)
                {
                    int latlonlength = indexLengthLat*indexLengthLon;
                    for (int i_empty=0; i_empty<cutStart; i_empty++)
                    {
                        for (int i_emptylatlon=0; i_emptylatlon<latlonlength; i_emptylatlon++)
                        {
                            data[indexBegining] = NaNFloat;
                            indexBegining++;
                        }
                    }
                }

                // Fill empty end with NaNs
                int indexEnd = indexLengthTime * indexLengthLat * indexLengthLon - 1;
                if(cutEnd>0)
                {
                    int latlonlength = indexLengthLat*indexLengthLon;
                    for (int i_empty=0; i_empty<cutEnd; i_empty++)
                    {
                        for (int i_emptylatlon=0; i_emptylatlon<latlonlength; i_emptylatlon++)
                        {
                            indexEnd++;
                            data[indexEnd] = NaNFloat;
                        }
                    }
                }

                // Get the indices for data
                size_t indexStartData4[4] = {0,0,0,0};
                size_t indexCountData4[4] = {0,0,0,0};
                ptrdiff_t indexStrideData4[4] = {0,0,0,0};
                size_t indexStartData3[3] = {0,0,0};
                size_t indexCountData3[3] = {0,0,0};
                ptrdiff_t indexStrideData3[3] = {0,0,0};

                if (nDims==4)
                {
                    // Set the indices for data
                    indexStartData4[0] = indexStartTime;
                    indexStartData4[1] = indexLevel;
                    indexStartData4[2] = indexStartLat;
                    indexStartData4[3] = indexStartLon;
                    indexCountData4[0] = indexLengthTime;
                    indexCountData4[1] = 1;
                    indexCountData4[2] = indexLengthLat;
                    indexCountData4[3] = indexLengthLon;
                    indexStrideData4[0] = indexStepTime;
                    indexStrideData4[1] = 1;
                    indexStrideData4[2] = indexStepLat;
                    indexStrideData4[3] = indexStepLon;

                    // In the netCDF Common Data Language, variables are printed with the outermost dimension first and the innermost dimension last.
                    ncFile.GetVarSample(dataName, indexStartData4, indexCountData4, indexStrideData4, &data[indexBegining]);
                }
                else
                {
                    // Set the indices for data
                    indexStartData3[0] = indexStartTime;
                    indexStartData3[1] = indexStartLat;
                    indexStartData3[2] = indexStartLon;
                    indexCountData3[0] = indexLengthTime;
                    indexCountData3[1] = indexLengthLat;
                    indexCountData3[2] = indexLengthLon;
                    indexStrideData3[0] = indexStepTime;
                    indexStrideData3[1] = indexStepLat;
                    indexStrideData3[2] = indexStepLon;

                    // In the netCDF Common Data Language, variables are printed with the outermost dimension first and the innermost dimension last.
                    ncFile.GetVarSample(dataName, indexStartData3, indexCountData3, indexStrideData3, &data[indexBegining]);
                }

                // Load data at lon = 360°
                if(load360)
                {
                    // Resize the arrays to store the new data
                    int totlength360 = indexLengthTimeArray * indexLengthLat * 1;
                    data360.resize(totlength360);

                    // Set the indices
                    indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], 360, 0.01f);
                    if(indexStartLon==asOUT_OF_RANGE)
                    {
                        // If not found, try with negative angles
                        indexStartLon = asTools::SortedArraySearch(&axisDataLon[0], &axisDataLon[axisDataLonLength-1], 0, 0.01f);
                    }

                    if (nDims==4)
                    {
                        indexStartData4[3] = indexStartLon;
                        indexCountData4[3] = 1;
                        indexStrideData4[3] = 1;
                    }
                    else
                    {
                        indexStartData3[2] = indexStartLon;
                        indexCountData3[2] = 1;
                        indexStrideData3[2] = 1;
                    }

                    // Fill empty begining with NaNs
                    int indexBegining = 0;
                    if(cutStart>0)
                    {
                        int latlonlength = indexLengthLat*indexLengthLon;
                        for (int i_empty=0; i_empty<cutStart; i_empty++)
                        {
                            for (int i_emptylatlon=0; i_emptylatlon<latlonlength; i_emptylatlon++)
                            {
                                data360[indexBegining] = NaNFloat;
                                indexBegining++;
                            }
                        }
                    }

                    // Fill empty end with NaNs
                    int indexEnd = (indexLengthTime-1) * (indexLengthLat-1) * (indexLengthLon-1);
                    if(cutEnd>0)
                    {
                        int latlonlength = indexLengthLat*indexLengthLon;
                        for (int i_empty=0; i_empty<cutEnd; i_empty++)
                        {
                            for (int i_emptylatlon=0; i_emptylatlon<latlonlength; i_emptylatlon++)
                            {
                                indexEnd++;
                                data360[indexEnd] = NaNFloat;
                            }
                        }
                    }

                    // Load data at 0° (corresponds to 360°)
                    if (nDims==4)
                    {
                        ncFile.GetVarSample(dataName, indexStartData4, indexCountData4, indexStrideData4, &data360[indexBegining]);
                    }
                    else
                    {
                        ncFile.GetVarSample(dataName, indexStartData3, indexCountData3, indexStrideData3, &data360[indexBegining]);
                    }
                }

                // Keep data for later treatment
                vectIndexLengthLat.push_back(indexLengthLat);
                vectIndexLengthLon.push_back(indexLengthLon);
                vectLoad360.push_back(load360);
                vectTotLength.push_back(totLength);
                vectData.push_back(data);
                vectData360.push_back(data360);
            }

            // Close the nc file
            ncFile.Close();

            ThreadsManager().CritSectionNetCDF().Leave();

            // The container for extracted data from every composite
            VVArray2DFloat compositeData;

            // Treat data
            for (int i_area = 0; i_area<iterationNb; i_area++)
            {
                // Extract data
                int indexLengthLat = vectIndexLengthLat[i_area];
                int indexLengthLon = vectIndexLengthLon[i_area];
                bool load360 = vectLoad360[i_area];
                int totLength = vectTotLength[i_area];
                VectorShort data = vectData[i_area];
                VectorShort data360 = vectData360[i_area];

                // Containers for results
                Array2DFloat latlonData(indexLengthLat,indexLengthLon);
                if(load360)
                {
                    latlonData.resize(indexLengthLat,indexLengthLon+1);
                }

                VArray2DFloat latlonTimeData;
                latlonTimeData.reserve(totLength);
                int ind = 0;
                VectorDouble valsNaNs = m_Catalog.GetNan();

                // Loop to extract the data from the array
                for (int i_time=0; i_time<indexLengthTimeArray; i_time++)
                {
                    for (int i_lat=0; i_lat<indexLengthLat; i_lat++)
                    {
                        for (int i_lon=0; i_lon<indexLengthLon; i_lon++)
                        {
                            ind = i_lon;
                            ind += i_lat * indexLengthLon;
                            ind += i_time * indexLengthLon * indexLengthLat;

                            if (scalingNeeded)
                            {
                                // Add the Offset and multiply by the Scale Factor
                                latlonData(i_lat,i_lon) = (float)data[ind] * dataScaleFactor + dataAddOffset;
                            }
                            else
                            {
                                latlonData(i_lat,i_lon) = (float)data[ind];
                            }

                            // Check if not NaN
                            bool notNan = true;
                            for (size_t i_nan=0; i_nan<valsNaNs.size(); i_nan++)
                            {
                                if ((float)data[ind]==valsNaNs[i_nan] || latlonData(i_lat,i_lon)==valsNaNs[i_nan])
                                {
                                    notNan = false;
                                }
                            }
                            if (!notNan)
                            {
                                latlonData(i_lat,i_lon) = NaNFloat;
                            }
                        }

                        if(load360)
                        {
                            ind = i_lat;
                            ind += i_time * indexLengthLat;

                            if (scalingNeeded)
                            {
                                // Add the Offset and multiply by the Scale Factor
                                latlonData(i_lat,indexLengthLon) = (float)data360[ind] * dataScaleFactor + dataAddOffset;
                            }
                            else
                            {
                                latlonData(i_lat,indexLengthLon) = (float)data360[ind];
                            }

                            // Check if not NaN
                            bool notNan = true;
                            for (size_t i_nan=0; i_nan<valsNaNs.size(); i_nan++)
                            {
                                if ((float)data360[ind]==valsNaNs[i_nan] || latlonData(i_lat,indexLengthLon)==valsNaNs[i_nan])
                                {
                                    notNan = false;
                                }
                            }
                            if (!notNan)
                            {
                                latlonData(i_lat,indexLengthLon) = NaNFloat;
                            }
                        }
                    }

                    latlonTimeData.push_back(latlonData);

                    if(load360)
                    {
                        latlonData.setZero(indexLengthLat,indexLengthLon+1);
                    }
                    else
                    {
                        latlonData.setZero(indexLengthLat,indexLengthLon);
                    }
                    counterTime++;
                }

                compositeData.push_back(latlonTimeData);
                latlonTimeData.clear();
                data.clear();
                data360.clear();
            }

            // Merge the composites into m_Data
            if (!MergeComposites(compositeData, dataArea))
            {
                wxDELETE(dataArea);
                return false;
            }
        }

        if (desiredArea)
        {
            // Interpolate the loaded data on the desired grid
            if (!InterpolateOnGrid(dataArea, desiredArea))
            {
                wxDELETE(dataArea);
                return false;
            }
        }

        // Check the time dimension
        int compositesNb = 1;
        if (desiredArea)
        {
            compositesNb = dataArea->GetNbComposites();
        }
        counterTime /= compositesNb;
        if (!CheckTimeLength(counterTime))
        {
            wxDELETE(dataArea);
            return false;
        }

        #ifndef UNIT_TESTING
            #if wxUSE_GUI
                progressBar.Destroy();
            #endif
        #endif

        wxDELETE(dataArea);
    }
    catch(bad_alloc& ba)
    {
        wxString msg(ba.what(), wxConvUTF8);
        asLogError(wxString::Format(_("Bad allocation caught when loading archive data: %s"), msg.c_str()));
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

bool asDataPredictorArchive::ClipToArea(asGeoAreaCompositeGrid *desiredArea)
{
    double Umin = desiredArea->GetAbsoluteUmin();
    double Umax = desiredArea->GetAbsoluteUmax();
    wxASSERT(m_AxisLon.size()>0);
    int UstartIndex = asTools::SortedArraySearch(&m_AxisLon[0], &m_AxisLon[m_AxisLon.size()-1], Umin);
    int UendIndex = asTools::SortedArraySearch(&m_AxisLon[0], &m_AxisLon[m_AxisLon.size()-1], Umax);
    if (UstartIndex<0)
    {
        UstartIndex = asTools::SortedArraySearch(&m_AxisLon[0], &m_AxisLon[m_AxisLon.size()-1], Umin+desiredArea->GetAxisUmax());
        UendIndex = asTools::SortedArraySearch(&m_AxisLon[0], &m_AxisLon[m_AxisLon.size()-1], Umax+desiredArea->GetAxisUmax());
        if (UstartIndex<0 || UendIndex<0)
        {
            asLogError(_("An error occured while trying to clip data to another area (extended axis)."));
            asLogError(wxString::Format(_("Looking for lon %.2f and %.2f inbetween %.2f to %.2f."),
                                        Umin+desiredArea->GetAxisUmax(), Umax+desiredArea->GetAxisUmax(), m_AxisLon[0], m_AxisLon[m_AxisLon.size()-1] ));
            return false;
        }
    }
    if (UstartIndex<0 || UendIndex<0)
    {

        asLogError(_("An error occured while trying to clip data to another area."));
        asLogError(wxString::Format(_("Looking for lon %.2f and %.2f inbetween %.2f to %.2f."),
                                    Umin, Umax, m_AxisLon[0], m_AxisLon[m_AxisLon.size()-1] ));
        return false;
    }
    int Ulength = UendIndex-UstartIndex+1;

    double Vmin = desiredArea->GetAbsoluteVmin();
    double Vmax = desiredArea->GetAbsoluteVmax();
    wxASSERT(m_AxisLat.size()>0);
    int VstartIndex = asTools::SortedArraySearch(&m_AxisLat[0], &m_AxisLat[m_AxisLat.size()-1], Vmin);
    int VendIndex = asTools::SortedArraySearch(&m_AxisLat[0], &m_AxisLat[m_AxisLat.size()-1], Vmax);
    if (UstartIndex<0)
    {
        VstartIndex = asTools::SortedArraySearch(&m_AxisLat[0], &m_AxisLat[m_AxisLat.size()-1], Vmin+desiredArea->GetAxisVmax());
        VendIndex = asTools::SortedArraySearch(&m_AxisLat[0], &m_AxisLat[m_AxisLat.size()-1], Vmax+desiredArea->GetAxisVmax());
        if (VstartIndex<0 || VendIndex<0)
        {
            asLogError(_("An error occured while trying to clip data to another area (extended axis)."));
            asLogError(wxString::Format(_("Looking for lat %.2f and %.2f inbetween %.2f to %.2f."),
                                        Vmin+desiredArea->GetAxisVmax(), Vmax+desiredArea->GetAxisVmax(), m_AxisLat[0], m_AxisLat[m_AxisLat.size()-1] ));
            return false;
        }
    }
    if (VstartIndex<0 || VendIndex<0)
    {
        asLogError(_("An error occured while trying to clip data to another area."));
        asLogError(wxString::Format(_("Looking for lat %.2f and %.2f inbetween %.2f to %.2f."),
                                    Vmin, Vmax, m_AxisLat[0], m_AxisLat[m_AxisLat.size()-1] ));
        return false;
    }

    int VstartIndexReal = wxMin(VstartIndex, VendIndex);
    int Vlength = abs(VendIndex-VstartIndex)+1;

    // Check if already the correct size
    if (VstartIndexReal==0 && UstartIndex==0 && Vlength==m_AxisLat.size() && Ulength==m_AxisLon.size() )
    {
        if (IsPreprocessed())
        {
            if(m_Data[0].cols()==m_AxisLon.size() && m_Data[0].rows()==2*m_AxisLat.size() )
            {
                // Nothing to do
                return true;
            }
            else
            {
                // Clear axes
                Array1DFloat newAxisLon(Ulength);
                for (int i=0; i<Ulength; i++)
                {
                    newAxisLon[i] = NaNFloat;
                }
                m_AxisLon = newAxisLon;

                Array1DFloat newAxisLat(2*Vlength);
                for (int i=0; i<2*Vlength; i++)
                {
                    newAxisLat[i] = NaNFloat;
                }
                m_AxisLat = newAxisLat;

                m_LatPtsnb = m_AxisLat.size();
                m_LonPtsnb = m_AxisLon.size();
            }
        }
        else
        {
            // Nothing to do
            return true;
        }
    }
    else
    {
        if (!CanBeClipped())
        {
            asLogError(_("The preprocessed area cannot be clipped to another area."));
            return false;
        }

        if (IsPreprocessed())
        {
            wxString method = GetPreprocessMethod();
            if (method.IsSameAs("MergeCouplesAndMultiply"))
            {
                VArray2DFloat originalData = m_Data;

                if(originalData[0].cols()!=m_AxisLon.size() || originalData[0].rows()!=2*m_AxisLat.size() )
                {
                    asLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    asLogError(wxString::Format("originalData[0].cols() = %d, m_AxisLon.size() = %d, originalData[0].rows() = %d, m_AxisLat.size() = %d", (int)originalData[0].cols(), (int)m_AxisLon.size(), (int)originalData[0].rows(), (int)m_AxisLat.size()));
                    return false;
                }

                for (unsigned int i=0; i<originalData.size(); i++)
                {
                    Array2DFloat dat1 = originalData[i].block(VstartIndexReal,UstartIndex,Vlength,Ulength);
                    Array2DFloat dat2 = originalData[i].block(VstartIndexReal+m_AxisLat.size(),UstartIndex,Vlength,Ulength);
                    Array2DFloat datMerged(2*Vlength, Ulength);
                    datMerged.block(0,0,Vlength,Ulength) = dat1;
                    datMerged.block(Vlength,0,Vlength,Ulength) = dat2;
                    m_Data[i] = datMerged;
                }

                Array1DFloat newAxisLon(Ulength);
                for (int i=0; i<Ulength; i++)
                {
                    newAxisLon[i] = NaNFloat;
                }
                m_AxisLon = newAxisLon;

                Array1DFloat newAxisLat(2*Vlength);
                for (int i=0; i<2*Vlength; i++)
                {
                    newAxisLat[i] = NaNFloat;
                }
                m_AxisLat = newAxisLat;

                m_LatPtsnb = m_AxisLat.size();
                m_LonPtsnb = m_AxisLon.size();

                return true;

            }
            else if (method.IsSameAs("HumidityFlux"))
            {
                VArray2DFloat originalData = m_Data;

                if(originalData[0].cols()!=m_AxisLon.size() || originalData[0].rows()!=2*m_AxisLat.size() )
                {
                    asLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    asLogError(wxString::Format("originalData[0].cols() = %d, m_AxisLon.size() = %d, originalData[0].rows() = %d, m_AxisLat.size() = %d", (int)originalData[0].cols(), (int)m_AxisLon.size(), (int)originalData[0].rows(), (int)m_AxisLat.size()));
                    return false;
                }

                for (unsigned int i=0; i<originalData.size(); i++)
                {
                    Array2DFloat dat1 = originalData[i].block(VstartIndexReal,UstartIndex,Vlength,Ulength);
                    Array2DFloat dat2 = originalData[i].block(VstartIndexReal+m_AxisLat.size(),UstartIndex,Vlength,Ulength);
                    Array2DFloat datMerged(2*Vlength, Ulength);
                    datMerged.block(0,0,Vlength,Ulength) = dat1;
                    datMerged.block(Vlength,0,Vlength,Ulength) = dat2;
                    m_Data[i] = datMerged;
                }

                Array1DFloat newAxisLon(Ulength);
                for (int i=0; i<Ulength; i++)
                {
                    newAxisLon[i] = NaNFloat;
                }
                m_AxisLon = newAxisLon;

                Array1DFloat newAxisLat(2*Vlength);
                for (int i=0; i<2*Vlength; i++)
                {
                    newAxisLat[i] = NaNFloat;
                }
                m_AxisLat = newAxisLat;

                m_LatPtsnb = m_AxisLat.size();
                m_LonPtsnb = m_AxisLon.size();

                return true;
            }
            else
            {
                asLogError(_("Wrong proprocessing definition (cannot be clipped to another area)."));
                return false;
            }
        }
    }

    VArray2DFloat originalData = m_Data;
    for (unsigned int i=0; i<originalData.size(); i++)
    {
        m_Data[i] = originalData[i].block(VstartIndexReal,UstartIndex,Vlength,Ulength);
    }

    Array1DFloat newAxisLon(Ulength);
    for (int i=0; i<Ulength; i++)
    {
        newAxisLon[i] = m_AxisLon[UstartIndex+i];
    }
    m_AxisLon = newAxisLon;

    Array1DFloat newAxisLat(Vlength);
    for (int i=0; i<Vlength; i++)
    {
        newAxisLat[i] = m_AxisLat[VstartIndexReal+i];
    }
    m_AxisLat = newAxisLat;

    m_LatPtsnb = m_AxisLat.size();
    m_LonPtsnb = m_AxisLon.size();

    return true;
}

bool asDataPredictorArchive::CheckTimeArray(asTimeArray &timeArray)
{
    if (!timeArray.IsSimpleMode())
    {
        asLogError(_("The data loading only accepts time arrays in simple mode."));
        return false;
    }

    // Check the time length
    if (timeArray.GetFirst()<m_Catalog.GetStart())
    {
        asLogError(_("The time array begins before the data start. operation canceled."));
        return false;
    }
    if (timeArray.GetLast()>m_Catalog.GetEnd())
    {
        asLogError(_("The time array ends after the data end. operation canceled."));
        return false;
    }

    // Check the time steps
    if ((timeArray.GetTimeStepDays()>0) && (m_Catalog.GetTimeStepDays()>timeArray.GetTimeStepDays()))
    {
        asLogError(_("The desired timestep is smaller than the data timestep."));
        return false;
    }
    double intpart, fractpart;
    fractpart = modf(timeArray.GetTimeStepDays()/m_Catalog.GetTimeStepDays(), &intpart);
    if (fractpart>0.0000001)
    {
        asLogError(_("The desired timestep is not a multiple of the data timestep."));
        return false;
    }
    fractpart = modf((timeArray.GetFirstDayHour()-m_Catalog.GetFirstTimeStepHours())/m_Catalog.GetTimeStepHours(), &intpart);
    if (fractpart>0.0000001)
    {
        asLogError(wxString::Format(_("The desired start (%gh) is not coherent with the data properties."), timeArray.GetFirstDayHour()));
        return false;
    }

    return true;
}
