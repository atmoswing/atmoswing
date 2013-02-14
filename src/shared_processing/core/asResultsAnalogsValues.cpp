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
 
#include "asResultsAnalogsValues.h"

#include "asFileNetcdf.h"


asResultsAnalogsValues::asResultsAnalogsValues()
:
asResults()
{
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Read("/IntermediateResults/SaveAnalogValues", &m_SaveIntermediateResults, false);
    wxFileConfig::Get()->Read("/IntermediateResults/LoadAnalogValues", &m_LoadIntermediateResults, false);
    ThreadsManager().CritSectionConfig().Leave();
}

asResultsAnalogsValues::~asResultsAnalogsValues()
{
    //dtor
}

void asResultsAnalogsValues::Init(asParameters &params)
{
    m_PredictandStationId = params.GetPredictandStationId();
    if(m_SaveIntermediateResults || m_LoadIntermediateResults) BuildFileName();

    // Resize to 0 to avoid keeping old results
    m_TargetDates.resize(0);
    m_TargetValuesNorm.resize(0);
    m_TargetValuesGross.resize(0);
    m_AnalogsCriteria.resize(0,0);
    m_AnalogsValuesNorm.resize(0,0);
    m_AnalogsValuesGross.resize(0,0);
}

void asResultsAnalogsValues::BuildFileName()
{
    ThreadsManager().CritSectionConfig().Enter();
    m_FilePath = wxFileConfig::Get()->Read("/StandardPaths/IntermediateResultsDir", asConfig::GetDefaultUserWorkingDir() + "IntermediateResults" + DS);
    ThreadsManager().CritSectionConfig().Leave();
    m_FilePath.Append(DS);
    m_FilePath.Append(wxString::Format("AnalogsValues_id%d_step%d", m_PredictandStationId, m_CurrentStep));
    m_FilePath.Append(".nc");
}

bool asResultsAnalogsValues::Save(const wxString &AlternateFilePath)
{
    // If we don't want to save, skip
    if(!m_SaveIntermediateResults) return false;
    wxString message = _("Saving intermediate file: ") + m_FilePath;
    asLogMessage(message);

    // Get the file path
    wxString ResultsFile;
    if (AlternateFilePath.IsEmpty())
    {
        ResultsFile = m_FilePath;
    }
    else
    {
        ResultsFile = AlternateFilePath;
    }

    // Get the elements size
    size_t Ntime = m_AnalogsCriteria.rows();
    size_t Nanalogs = m_AnalogsCriteria.cols();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(ResultsFile, asFileNetcdf::Replace);
    if(!ncFile.Open())
    {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Define dimensions. Time is the unlimited dimension.
    ncFile.DefDim("time");
    ncFile.DefDim("analogs", Nanalogs);

    // The dimensions name array is used to pass the dimensions to the variable.
    VectorStdString DimNames1;
    DimNames1.push_back("time");
    VectorStdString DimNames2;
    DimNames2.push_back("time");
    DimNames2.push_back("analogs");

    // Define variables: the analogcriteria and the corresponding dates
    ncFile.DefVar("targetdates", NC_FLOAT, 1, DimNames1);
    ncFile.DefVar("targetvaluesnorm", NC_FLOAT, 1, DimNames1);
    ncFile.DefVar("targetvaluesgross", NC_FLOAT, 1, DimNames1);
    ncFile.DefVar("analogscriteria", NC_FLOAT, 2, DimNames2);
    ncFile.DefVar("analogsvaluesnorm", NC_FLOAT, 2, DimNames2);
    ncFile.DefVar("analogsvaluesgross", NC_FLOAT, 2, DimNames2);

    // Put attributes
    DefTargetDatesAttributes(ncFile);
    DefTargetValuesNormAttributes(ncFile);
    DefTargetValuesGrossAttributes(ncFile);
    DefAnalogsCriteriaAttributes(ncFile);
    DefAnalogsValuesNormAttributes(ncFile);
    DefAnalogsValuesGrossAttributes(ncFile);

    // End definitions: leave define mode
    ncFile.EndDef();

    // Provide sizes for variables
    size_t start1D[] = {0};
    size_t count1D[] = {Ntime};
    size_t start2D[] = {0, 0};
    size_t count2D[] = {Ntime, Nanalogs};

    // Set the matrices in vectors
    int totLength = Ntime * Nanalogs;
    VectorFloat analogsCriteria(totLength);
    VectorFloat analogsValuesNorm(totLength);
    VectorFloat analogsValuesGross(totLength);
    int ind = 0;

    for (unsigned int i_time=0; i_time<Ntime; i_time++)
    {
        for (unsigned int i_analog=0; i_analog<Nanalogs;i_analog++)
        {
            ind = i_analog;
            ind += i_time * Nanalogs;
            analogsCriteria[ind] = m_AnalogsCriteria(i_time,i_analog);
            analogsValuesNorm[ind] = m_AnalogsValuesNorm(i_time,i_analog);
            analogsValuesGross[ind] = m_AnalogsValuesGross(i_time,i_analog);
        }
    }

    // Write data
    ncFile.PutVarArray("targetdates", start1D, count1D, &m_TargetDates(0));
    ncFile.PutVarArray("targetvaluesnorm", start1D, count1D, &m_TargetValuesNorm(0));
    ncFile.PutVarArray("targetvaluesgross", start1D, count1D, &m_TargetValuesGross(0));
    ncFile.PutVarArray("analogscriteria", start2D, count2D, &analogsCriteria[0]);
    ncFile.PutVarArray("analogsvaluesnorm", start2D, count2D, &analogsValuesNorm[0]);
    ncFile.PutVarArray("analogsvaluesgross", start2D, count2D, &analogsValuesGross[0]);

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}

bool asResultsAnalogsValues::Load(const wxString &AlternateFilePath)
{
    // If we don't want to save or the file doesn't exist
    if(!m_LoadIntermediateResults) return false;
    if(!Exists()) return false;
    if(m_CurrentStep!=0) return false;

    // Get the file path
    wxString ResultsFile;
    if (AlternateFilePath.IsEmpty())
    {
        ResultsFile = m_FilePath;
    }
    else
    {
        ResultsFile = AlternateFilePath;
    }

    ThreadsManager().CritSectionNetCDF().Enter();

    // Open the NetCDF file
    asFileNetcdf ncFile(ResultsFile, asFileNetcdf::ReadOnly);
    if(!ncFile.Open())
    {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Get the elements size
    int Ntime = ncFile.GetDimLength("time");
    int Nanalogs = ncFile.GetDimLength("analogs");

    // Get time
    m_TargetDates.resize( Ntime );
    ncFile.GetVar("targetdates", &m_TargetDates[0]);

    // Create vectors for matrices data
    int totLength = Ntime * Nanalogs;
    VectorFloat analogsCriteria(totLength);
    VectorFloat analogsValuesNorm(totLength);
    VectorFloat analogsValuesGross(totLength);

    // Get data
    m_TargetValuesNorm.resize( Ntime );
    ncFile.GetVar("targetvaluesnorm", &m_TargetValuesNorm[0]);
    m_TargetValuesGross.resize( Ntime );
    ncFile.GetVar("targetvaluesgross", &m_TargetValuesGross[0]);
    size_t IndexStart[2] = {0,0};
    size_t IndexCount[2] = {Ntime, Nanalogs};
    ncFile.GetVarArray("analogscriteria", IndexStart, IndexCount, &analogsCriteria[0]);
    ncFile.GetVarArray("analogsvaluesnorm", IndexStart, IndexCount, &analogsValuesNorm[0]);
    ncFile.GetVarArray("analogsvaluesgross", IndexStart, IndexCount, &analogsValuesGross[0]);

    // Set data into the matrices
    m_AnalogsValuesNorm.resize( Ntime, Nanalogs );
    m_AnalogsValuesGross.resize( Ntime, Nanalogs );
    m_AnalogsCriteria.resize( Ntime, Nanalogs );
    int ind = 0;
    for (int i_time=0; i_time<Ntime; i_time++)
    {
        for (int i_analog=0; i_analog<Nanalogs;i_analog++)
        {
            ind = i_analog;
            ind += i_time * Nanalogs;
            m_AnalogsCriteria(i_time,i_analog) = analogsCriteria[ind];
            m_AnalogsValuesNorm(i_time,i_analog) = analogsValuesNorm[ind];
            m_AnalogsValuesGross(i_time,i_analog) = analogsValuesGross[ind];
        }
    }

    ThreadsManager().CritSectionNetCDF().Leave();

    ncFile.Close();

    return true;
}
