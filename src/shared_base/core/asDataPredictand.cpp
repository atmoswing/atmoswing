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
 
#include "asDataPredictand.h"

#include <asCatalogPredictands.h>
#include <asFileDat.h>
#include <asTimeArray.h>
#include <asDataPredictandPrecipitation.h>
#include <asDataPredictandLightnings.h>


asDataPredictand::asDataPredictand(PredictandDB predictandDB)
{
    m_PredictandDB = predictandDB;
    m_FileVersion = 1.0;
}

asDataPredictand::~asDataPredictand()
{
    //dtor
}

asDataPredictand* asDataPredictand::GetInstance(const wxString& PredictandDBStr)
{
    if (PredictandDBStr.CmpNoCase("StationsDailyPrecipitation")==0)
    {
        asDataPredictand* db = new asDataPredictandPrecipitation(StationsDailyPrecipitation);
        return db;
    }
    else if (PredictandDBStr.CmpNoCase("Stations6HourlyPrecipitation")==0)
    {
        asDataPredictand* db = new asDataPredictandPrecipitation(Stations6HourlyPrecipitation);
        return db;
    }
    else if (PredictandDBStr.CmpNoCase("Stations6HourlyOfDailyPrecipitation")==0)
    {
        asDataPredictand* db = new asDataPredictandPrecipitation(Stations6HourlyOfDailyPrecipitation);
        return db;
    }
    else if (PredictandDBStr.CmpNoCase("RegionalDailyPrecipitation")==0)
    {
        asDataPredictand* db = new asDataPredictandPrecipitation(RegionalDailyPrecipitation);
        return db;
    }
    else if (PredictandDBStr.CmpNoCase("ResearchDailyPrecipitation")==0)
    {
        asDataPredictand* db = new asDataPredictandPrecipitation(ResearchDailyPrecipitation);
        return db;
    }
    else if (PredictandDBStr.CmpNoCase("StationsDailyLightnings")==0)
    {
        asDataPredictand* db = new asDataPredictandLightnings(StationsDailyLightnings);
        return db;
    }
    else
    {
        asLogError(wxString::Format(_("The given predictand DB type (%s) in unknown"), PredictandDBStr.c_str()));
    }

    return NULL;
}

bool asDataPredictand::InitMembers(const wxString &AlternateFilePath)
{
    // Starting and ending date of the DB, to be overwritten
    m_DateStart = asTime::GetMJD(2100,1,1);
    m_DateEnd = asTime::GetMJD(1800,1,1);

    // Get the datasets IDs
    asCatalog::DatasetIdList datsetList = asCatalog::GetDatasetIdList(Predictand, AlternateFilePath);

    // Look for final matrix dimension
    m_StationsNb = 0;
    for (size_t i_set=0; i_set<datsetList.Id.size(); i_set++)
    {
        // Get the catalog information
        asCatalogPredictands catalog(AlternateFilePath);
        if(!catalog.Load(datsetList.Id[i_set])) return false;

        // Check if the dataset should be included in the DB
        bool includeinDB = false;
        VectorPredictandDB datasetDBs = catalog.GetDBInclude();
        for (size_t i_db=0; i_db<datasetDBs.size(); i_db++)
        {
            if (m_PredictandDB==datasetDBs[i_db]) includeinDB = true;
        }

        //Include in DB
        if (includeinDB)
        {
            // Get first and last date
            if (catalog.GetStart()<m_DateStart)
                m_DateStart = catalog.GetStart();
            if (catalog.GetEnd()>m_DateEnd)
                m_DateEnd = catalog.GetEnd();

            // Get the number of stations
            asCatalog::DataIdListInt datListCheck = asCatalog::GetDataIdListInt(Predictand, datsetList.Id[i_set], AlternateFilePath);
            m_StationsNb += datListCheck.Id.size();
        }
    }

    // Get the timestep
    switch (m_PredictandDB)
    {
        case (StationsDailyPrecipitation):
        case (RegionalDailyPrecipitation):
        case (ResearchDailyPrecipitation):
        case (StationsDailyLightnings):
            m_TimeStepDays = 1;
            break;
        case (Stations6HourlyPrecipitation):
        case (Stations6HourlyOfDailyPrecipitation):
            m_TimeStepDays = 6.0/24.0;
            break;
        default:
            wxFAIL;
            m_TimeStepDays = 0;
    }

    // Get the time length
    m_TimeLength = ((m_DateEnd-m_DateStart) / m_TimeStepDays) + 1;

    return true;
}

bool asDataPredictand::InitBaseContainers()
{
    if (m_StationsNb<1)
    {
        asLogError(_("The stations number is inferior to 1."));
        return false;
    }
    if (m_TimeLength<1)
    {
        asLogError(_("The time length is inferior to 1."));
        return false;
    }
    m_StationsName.resize(m_StationsNb);
    m_StationsIds.resize(m_StationsNb);
    m_StationsLocCoordU.resize(m_StationsNb);
    m_StationsLocCoordV.resize(m_StationsNb);
    m_StationsLon.resize(m_StationsNb);
    m_StationsLat.resize(m_StationsNb);
    m_StationsHeight.resize(m_StationsNb);
    m_StationsStart.resize(m_StationsNb);
    m_StationsEnd.resize(m_StationsNb);
    m_Time.resize(m_TimeLength);
    m_DataGross.resize(m_TimeLength, m_StationsNb);
    m_DataGross.fill(NaNFloat);
    m_DataNormalized.resize(m_TimeLength, m_StationsNb);
    m_DataNormalized.fill(NaNFloat);

    return true;
}

bool asDataPredictand::IncludeInDB(const wxString &datasetID, const wxString &AlternateFilePath)
{
    // Get some dataset information
    asCatalogPredictands currentDataset(AlternateFilePath);
    if(!currentDataset.Load(datasetID)) return false;

    // Check if the dataset should be included in the DB
    VectorPredictandDB datasetDBs = currentDataset.GetDBInclude();
    for (size_t i_db=0; i_db<datasetDBs.size(); i_db++)
    {
        if (m_PredictandDB==datasetDBs[i_db]) return true;
    }
    return false;
}

bool asDataPredictand::SetStationProperties(asCatalogPredictands &currentData, size_t stationIndex)
{
    m_StationsName[stationIndex] = currentData.GetStationName();
    m_StationsIds(stationIndex) = currentData.GetStationId();
    m_StationsLocCoordU(stationIndex) = currentData.GetStationCoord().u;
    m_StationsLocCoordV(stationIndex) = currentData.GetStationCoord().v;
// FIXME (Pascal#1#): Implement lon/lat
    m_StationsLon(stationIndex) = NaNDouble;
    m_StationsLat(stationIndex) = NaNDouble;
    m_StationsHeight(stationIndex) = currentData.GetStationHeight();
    m_StationsStart(stationIndex) = currentData.GetStationStart();
    m_StationsEnd(stationIndex) = currentData.GetStationEnd();

    return true;
}

bool asDataPredictand::GetFileContent(asCatalogPredictands &currentData, size_t stationIndex, const wxString &AlternateDataDir, const wxString &AlternatePatternDir)
{
    // Load file
    wxString fileFullPath;
    if (!AlternateDataDir.IsEmpty())
    {
        fileFullPath = AlternateDataDir + DS + currentData.GetStationFilename();
    }
    else
    {
        fileFullPath = currentData.GetDataPath() + currentData.GetStationFilename();
    }
    asFileDat datFile(fileFullPath, asFile::ReadOnly);
    if(!datFile.Open()) return false;

    // Get the parsing format
    wxString stationFilePattern = currentData.GetStationFilepattern();
    asFileDat::Pattern filePattern = asFileDat::GetPattern(stationFilePattern, AlternatePatternDir);
    size_t maxCharWidth = asFileDat::GetPatternLineMaxCharWidth(filePattern);

    // Jump the header
    datFile.SkipLines(filePattern.HeaderLines);

    // Get first index on the tima axis
    int startIndex = asTools::SortedArraySearch(&m_Time[0], &m_Time[m_Time.size()-1], currentData.GetStationStart());
    if (startIndex==asOUT_OF_RANGE || startIndex==asNOT_FOUND)
    {
        asLogError(wxString::Format(_("The given start date for \"%s\" is out of the catalog range."), currentData.GetStationName().c_str()));
        return false;
    }

    int timeIndex = startIndex;

    // Parse every line until the end of the file
    while (!datFile.EndOfFile())
    {
        // Get current line
        wxString lineContent = datFile.GetLineContent();

        // Check the line width
        if (lineContent.Len()>=maxCharWidth)
        {
            // Check the size of the array
            if(timeIndex>=m_TimeLength)
            {
                asLogError(wxString::Format(_("The time index is larger than the matrix (timeIndex = %d, m_TimeLength = %d)."), (int)timeIndex, (int)m_TimeLength));
                return false;
            }

            switch (filePattern.StructType)
            {
                case (asFileDat::ConstantWidth):
                {
                    if(filePattern.ParseTime)
                    {
                        // Containers. Must be a double to use wxString::ToDouble
                        double valTimeYear=0, valTimeMonth=0, valTimeDay=0, valTimeHour=0, valTimeMinute=0;

                        // Get time value
                        if (filePattern.TimeYearBegin!=0 && filePattern.TimeYearEnd!=0 && filePattern.TimeMonthBegin!=0 && filePattern.TimeMonthEnd!=0 && filePattern.TimeDayBegin!=0 && filePattern.TimeDayEnd!=0)
                        {
                            lineContent.Mid(filePattern.TimeYearBegin-1, filePattern.TimeYearEnd-filePattern.TimeYearBegin+1).ToDouble(&valTimeYear);
                            lineContent.Mid(filePattern.TimeMonthBegin-1, filePattern.TimeMonthEnd-filePattern.TimeMonthBegin+1).ToDouble(&valTimeMonth);
                            lineContent.Mid(filePattern.TimeDayBegin-1, filePattern.TimeDayEnd-filePattern.TimeDayBegin+1).ToDouble(&valTimeDay);
                        } else {
                            asLogError(_("The data file pattern is not correctly defined."));
                            return false;
                        }

                        if (filePattern.TimeHourBegin!=0 && filePattern.TimeHourEnd!=0)
                        {
                            lineContent.Mid(filePattern.TimeHourBegin-1, filePattern.TimeHourEnd-filePattern.TimeHourBegin+1).ToDouble(&valTimeHour);
                        }
                        if (filePattern.TimeMinuteBegin!=0 && filePattern.TimeMinuteEnd!=0)
                        {
                            lineContent.Mid(filePattern.TimeMinuteBegin-1, filePattern.TimeMinuteEnd-filePattern.TimeMinuteBegin+1).ToDouble(&valTimeMinute);
                        }

                        double dateData = asTime::GetMJD(valTimeYear, valTimeMonth, valTimeDay, valTimeHour, valTimeMinute, 0);

                        // Check again date vector
                        if ( abs(dateData - m_Time(timeIndex)) > 0.0001)
                        {
                            wxString errorMessage = wxString::Format(_("Value in data : %6.4f (%s), value in time array : %6.4f (%s). In file %s"), dateData, asTime::GetStringTime(dateData,"DD.MM.YYYY").c_str(), m_Time(timeIndex), asTime::GetStringTime(m_Time(timeIndex),"DD.MM.YYYY").c_str(), currentData.GetStationFilename().c_str());
                            asLogError(wxString::Format(_("The time value doesn't match: %s"), errorMessage.c_str() ));
                            return false;
                        }
                    }

                    // Get Precipitation value
                    double valPrecipitationGross=0;
                    lineContent.Mid(filePattern.DataBegin-1, filePattern.DataEnd-filePattern.DataBegin+1).ToDouble(&valPrecipitationGross);

                    // Check if not NaN and store
                    bool notNan = true;
                    for (size_t i_nan=0; i_nan<currentData.GetNan().size(); i_nan++)
                    {
                        if (valPrecipitationGross==currentData.GetNan()[i_nan]) notNan = false;
                    }
                    if (notNan)
                    {
                        // Put value in the matrix
                        m_DataGross(timeIndex,stationIndex) = valPrecipitationGross;
                    }
                    timeIndex++;
                    break;
                }

                case (asFileDat::TabsDelimited):
                {
                    // Parse into a vector
                    VectorString vColumns;
                    wxString tmpLineContent = lineContent;
                    while( tmpLineContent.Find("\t") != wxNOT_FOUND )
                    {
                        int foundCol = tmpLineContent.Find("\t");
                        vColumns.push_back(tmpLineContent.Mid(0,foundCol));
                        tmpLineContent = tmpLineContent.Mid(foundCol+1);
                    }
                    if (!tmpLineContent.IsEmpty())
                    {
                        vColumns.push_back(tmpLineContent);
                    }

                    if(filePattern.ParseTime)
                    {
                        // Containers. Must be a double to use wxString::ToDouble
                        double valTimeYear=0, valTimeMonth=0, valTimeDay=0, valTimeHour=0, valTimeMinute=0;

                        // Get time value
                        if (filePattern.TimeYearBegin!=0 && filePattern.TimeMonthBegin!=0 && filePattern.TimeDayBegin!=0)
                        {
                            if ((unsigned)filePattern.TimeYearBegin>vColumns.size() || (unsigned)filePattern.TimeMonthBegin>vColumns.size() || (unsigned)filePattern.TimeDayBegin>vColumns.size())
                            {
                                asLogError(_("The data file pattern is not correctly defined. Trying to access an element (date) after the line width."));
                                return false;
                            }
                            vColumns[filePattern.TimeYearBegin-1].ToDouble(&valTimeYear);
                            vColumns[filePattern.TimeMonthBegin-1].ToDouble(&valTimeMonth);
                            vColumns[filePattern.TimeDayBegin-1].ToDouble(&valTimeDay);
                        } else {
                            asLogError(_("The data file pattern is not correctly defined."));
                            return false;
                        }

                        if (filePattern.TimeHourBegin!=0)
                        {
                            if ((unsigned)filePattern.TimeHourBegin>vColumns.size())
                            {
                                asLogError(_("The data file pattern is not correctly defined. Trying to access an element (hour) after the line width."));
                                return false;
                            }
                            vColumns[filePattern.TimeHourBegin-1].ToDouble(&valTimeHour);
                        }
                        if (filePattern.TimeMinuteBegin!=0)
                        {
                            if ((unsigned)filePattern.TimeMinuteBegin>vColumns.size())
                            {
                                asLogError(_("The data file pattern is not correctly defined. Trying to access an element (minute) after the line width."));
                                return false;
                            }
                            vColumns[filePattern.TimeMinuteBegin-1].ToDouble(&valTimeMinute);
                        }

                        double dateData = asTime::GetMJD(valTimeYear, valTimeMonth, valTimeDay, valTimeHour, valTimeMinute, 0);

                        // Check again date vector
                        if ( abs(dateData - m_Time(timeIndex)) > 0.001)
                        {
                            wxString errorMessage = wxString::Format(_("Value in data : %6.4f (%s), value in time array : %6.4f (%s). In file %s"), dateData, asTime::GetStringTime(dateData,"DD.MM.YYYY").c_str(), m_Time(timeIndex), asTime::GetStringTime(m_Time(timeIndex),"DD.MM.YYYY").c_str(), currentData.GetStationFilename().c_str());
                            asLogError(wxString::Format(_("The time value doesn't match: %s"), errorMessage.c_str() ));
                            return false;
                        }
                    }

                    // Get Precipitation value
                    double valPrecipitationGross=0;
                    vColumns[filePattern.DataBegin-1].ToDouble(&valPrecipitationGross);

                    // Check if not NaN and store
                    bool notNan = true;
                    for (size_t i_nan=0; i_nan<currentData.GetNan().size(); i_nan++)
                    {
                        if (valPrecipitationGross==currentData.GetNan()[i_nan]) notNan = false;
                    }
                    if (notNan)
                    {
                        // Put value in the matrix
                        m_DataGross(timeIndex,stationIndex) = valPrecipitationGross;
                    }
                    timeIndex++;
                    break;
                }
            }
        }
        else
        {
            if(lineContent.Len()>1)
            {
                asLogError(_("The line length doesn't match."));
                return false;
            }
        }
    }
    datFile.Close();

    // Get end index
    int endIndex = asTools::SortedArraySearch(&m_Time[0], &m_Time[m_Time.size()-1], currentData.GetStationEnd());
    if (endIndex==asOUT_OF_RANGE || endIndex==asNOT_FOUND)
    {
        asLogError(wxString::Format(_("The given end date for \"%s\" is out of the catalog range."), currentData.GetStationName().c_str()));
        return false;
    }

    // Check time width
    if (endIndex-startIndex!=timeIndex-startIndex-1)
    {
        wxString messageTime = wxString::Format(_("The length of the data in \"%s / %s\" is not coherent"), currentData.GetName().c_str(), currentData.GetStationName().c_str());
        asLogError(messageTime);
        return false;
    }

    return true;
}

Array2DFloat asDataPredictand::GetAnnualMax(double timeStepDays, int nansNbMax)
{
    // Flag to check the need of aggregation (timeStepDays>m_TimeStepDays)
    bool aggregate = false;
    int indexTimeSpanUp = 0;
    int indexTimeSpanDown = 0;

    if(timeStepDays==m_TimeStepDays)
    {
        aggregate = false;
    }
    else if(timeStepDays>m_TimeStepDays)
    {
        if(fmod(timeStepDays,m_TimeStepDays)>0.0000001)
        {
            asLogError(_("The timestep for the extraction of the predictands maximums has to be a multiple of the data timestep."));
            Array2DFloat emptyMatrix;
            emptyMatrix << NaNFloat;
            return emptyMatrix;
        }

        // Aggragation necessary
        aggregate = true;

        // indices to add or substract around the mid value
        indexTimeSpanUp = floor((timeStepDays/m_TimeStepDays)/2);
        indexTimeSpanDown = ceil((timeStepDays/m_TimeStepDays)/2)-1;
    }
    else
    {
        asLogError(_("The timestep for the extraction of the predictands maximums cannot be lower than the data timestep."));
        Array2DFloat emptyMatrix;
        emptyMatrix << NaNFloat;
        return emptyMatrix;
    }

    // Keep the real indices of years
    int indYearStart = 0;
    int indYearEnd = 0;

    // Get catalog beginning and end
    int yearStart = asTime::GetYear(m_DateStart);
    if (asTime::GetMonth(m_DateStart)!=1 || asTime::GetDay(m_DateStart)!=1)
    {
        yearStart++;
        indYearStart++;
    }
    int yearEnd = asTime::GetYear(m_DateEnd);
    indYearEnd = yearEnd-yearStart+indYearStart;
    if (asTime::GetMonth(m_DateEnd)!=12 || asTime::GetDay(m_DateEnd)!=31)
    {
        yearEnd--;
    }

    // Create the container
    Array2DFloat maxMatrix = Array2DFloat::Constant(m_StationsNb, indYearEnd+1, NaNFloat);

    // Look for maximums
    for (int i_stnb=0; i_stnb<m_StationsNb; i_stnb++)
    {
        for (int i_year=yearStart; i_year<=yearEnd; i_year++)
        {
            // The maximum value and a flag for accepted NaNs
            float annualmax = -99999;
            int nansNb = 0;

            // Find begining and end of the year
            int rowstart = asTools::SortedArraySearchFloor(&m_Time[0], &m_Time[m_TimeLength-1], asTime::GetMJD(i_year, 1, 1), asHIDE_WARNINGS );
            int rowend = asTools::SortedArraySearchFloor(&m_Time[0], &m_Time[m_TimeLength-1], asTime::GetMJD(i_year, 12, 31, 59, 59), asHIDE_WARNINGS);
            if ( (rowend==asOUT_OF_RANGE) | (rowend==asNOT_FOUND) )
            {
                if (i_year==yearEnd)
                {
                    rowend = m_TimeLength-1;
                }
                else
                {
                    annualmax = NaNFloat;
                }
            }
            rowend -= 1;

            // Get max
            if(!aggregate)
            {
                for (int i_row=rowstart; i_row<=rowend; i_row++)
                {
                    if (!asTools::IsNaN(m_DataGross(i_row, i_stnb)))
                    {
                        annualmax = wxMax(m_DataGross(i_row, i_stnb),annualmax);
                    }
                    else
                    {
                        nansNb++;
                    }
                }
                if (nansNb>nansNbMax)
                {
                    annualmax = NaNFloat;
                }
            }
            else
            {
                // Correction for both extremes
                rowstart = wxMax(rowstart-indexTimeSpanDown, 0);
                rowstart += indexTimeSpanDown;
                rowend = wxMin(rowend+indexTimeSpanUp, (int)m_DataGross.rows()-1);
                rowend -= indexTimeSpanUp;

                // Loop within the new limits
                for (int i_row=rowstart; i_row<=rowend; i_row++)
                {
                    float timeStepSum = 0;
                    for (int i_element=i_row-indexTimeSpanDown; i_element<=i_row+indexTimeSpanUp; i_element++)
                    {
                        if (!asTools::IsNaN(m_DataGross(i_element, i_stnb)))
                        {
                            timeStepSum += m_DataGross(i_element, i_stnb);
                        }
                        else
                        {
                            timeStepSum = NaNFloat;
                            break;
                        }
                    }

                    if (!asTools::IsNaN(timeStepSum))
                    {
                        annualmax = wxMax(timeStepSum,annualmax);
                    }
                    else
                    {
                        nansNb++;
                    }
                }
                if (nansNb>nansNbMax)
                {
                    annualmax = NaNFloat;
                }
            }

            maxMatrix(i_stnb, i_year-yearStart+indYearStart) = annualmax;
        }
    }

    return maxMatrix;
}

int asDataPredictand::GetStationIndex(int stationId)
{
    return asTools::SortedArraySearch(&m_StationsIds[0], &m_StationsIds[m_StationsNb-1], stationId);
}
