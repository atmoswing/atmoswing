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

#include "asResultsAnalogsDates.h"

#include "asFileNetcdf.h"


asResultsAnalogsDates::asResultsAnalogsDates()
        : asResults()
{
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    bool saveAnalogDatesStep1;
    pConfig->Read("/Optimizer/IntermediateResults/SaveAnalogDatesStep1", &saveAnalogDatesStep1, false);
    bool saveAnalogDatesStep2;
    pConfig->Read("/Optimizer/IntermediateResults/SaveAnalogDatesStep2", &saveAnalogDatesStep2, false);
    bool saveAnalogDatesStep3;
    pConfig->Read("/Optimizer/IntermediateResults/SaveAnalogDatesStep3", &saveAnalogDatesStep3, false);
    bool saveAnalogDatesStep4;
    pConfig->Read("/Optimizer/IntermediateResults/SaveAnalogDatesStep4", &saveAnalogDatesStep4, false);
    bool saveAnalogDatesAllSteps;
    pConfig->Read("/Optimizer/IntermediateResults/SaveAnalogDatesAllSteps", &saveAnalogDatesAllSteps, false);
    if (saveAnalogDatesStep1 || saveAnalogDatesStep2 || saveAnalogDatesStep3 || saveAnalogDatesStep4 ||
        saveAnalogDatesAllSteps) {
        m_saveIntermediateResults = true;
    }
    bool loadAnalogDatesStep1;
    pConfig->Read("/Optimizer/IntermediateResults/LoadAnalogDatesStep1", &loadAnalogDatesStep1, false);
    bool loadAnalogDatesStep2;
    pConfig->Read("/Optimizer/IntermediateResults/LoadAnalogDatesStep2", &loadAnalogDatesStep2, false);
    bool loadAnalogDatesStep3;
    pConfig->Read("/Optimizer/IntermediateResults/LoadAnalogDatesStep3", &loadAnalogDatesStep3, false);
    bool loadAnalogDatesStep4;
    pConfig->Read("/Optimizer/IntermediateResults/LoadAnalogDatesStep4", &loadAnalogDatesStep4, false);
    bool loadAnalogDatesAllSteps;
    pConfig->Read("/Optimizer/IntermediateResults/LoadAnalogDatesAllSteps", &loadAnalogDatesAllSteps, false);
    if (loadAnalogDatesStep1 || loadAnalogDatesStep2 || loadAnalogDatesStep3 || loadAnalogDatesStep4 ||
        loadAnalogDatesAllSteps) {
        m_loadIntermediateResults = true;
    }
    ThreadsManager().CritSectionConfig().Leave();
}

asResultsAnalogsDates::~asResultsAnalogsDates()
{
    //dtor
}

void asResultsAnalogsDates::Init(asParameters &params)
{
    m_predictandStationIds = params.GetPredictandStationIds();
    if (m_saveIntermediateResults || m_loadIntermediateResults)
        BuildFileName();

    // Resize to 0 to avoid keeping old results
    m_targetDates.resize(0);
    m_analogsCriteria.resize(0, 0);
    m_analogsDates.resize(0, 0);
}

void asResultsAnalogsDates::BuildFileName()
{
    ThreadsManager().CritSectionConfig().Enter();
    m_filePath = wxFileConfig::Get()->Read("/Paths/IntermediateResultsDir",
                                           asConfig::GetDefaultUserWorkingDir() + "IntermediateResults" + DS);
    ThreadsManager().CritSectionConfig().Leave();
    m_filePath.Append(DS);
    m_filePath.Append(wxString::Format("AnalogsDates_id_%s_step_%d", GetPredictandStationIdsList(), m_currentStep));
    m_filePath.Append(".nc");
}

bool asResultsAnalogsDates::Save(const wxString &AlternateFilePath) const
{
    // If we don't want to save, skip
    if (!m_saveIntermediateResults) {
        return false;
    } else {
        // Check if the current step is concerned
        ThreadsManager().CritSectionConfig().Enter();
        wxConfigBase *pConfig = wxFileConfig::Get();
        bool saveAnalogDatesStep1;
        pConfig->Read("/Optimizer/IntermediateResults/SaveAnalogDatesStep1", &saveAnalogDatesStep1, false);
        bool saveAnalogDatesStep2;
        pConfig->Read("/Optimizer/IntermediateResults/SaveAnalogDatesStep2", &saveAnalogDatesStep2, false);
        bool saveAnalogDatesStep3;
        pConfig->Read("/Optimizer/IntermediateResults/SaveAnalogDatesStep3", &saveAnalogDatesStep3, false);
        bool saveAnalogDatesStep4;
        pConfig->Read("/Optimizer/IntermediateResults/SaveAnalogDatesStep4", &saveAnalogDatesStep4, false);
        bool saveAnalogDatesAllSteps;
        pConfig->Read("/Optimizer/IntermediateResults/SaveAnalogDatesAllSteps", &saveAnalogDatesAllSteps, false);
        ThreadsManager().CritSectionConfig().Leave();

        if (!saveAnalogDatesAllSteps) {
            switch (m_currentStep) {
                case 0: {
                    if (!saveAnalogDatesStep1)
                        return false;
                    break;
                }
                case 1: {
                    if (!saveAnalogDatesStep2)
                        return false;
                    break;
                }
                case 2: {
                    if (!saveAnalogDatesStep3)
                        return false;
                    break;
                }
                case 3: {
                    if (!saveAnalogDatesStep4)
                        return false;
                    break;
                }
                default: {
                    return false;
                }
            }
        }
    }
    wxString message = _("Saving intermediate file: ") + m_filePath;
    asLogMessage(message);

    // Get the file path
    wxString ResultsFile;
    if (AlternateFilePath.IsEmpty()) {
        ResultsFile = m_filePath;
    } else {
        ResultsFile = AlternateFilePath;
    }

    // Get the elements size
    size_t Ntime = m_analogsCriteria.rows();
    size_t Nanalogs = m_analogsCriteria.cols();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(ResultsFile, asFileNetcdf::Replace);
    if (!ncFile.Open()) {
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
    ncFile.DefVar("target_dates", NC_FLOAT, 1, DimNames1);
    ncFile.DefVar("analog_criteria", NC_FLOAT, 2, DimNames2);
    ncFile.DefVar("analog_dates", NC_FLOAT, 2, DimNames2);

    // Put attributes
    DefTargetDatesAttributes(ncFile);
    DefAnalogsCriteriaAttributes(ncFile);
    DefAnalogsDatesAttributes(ncFile);

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
    VectorFloat analogsDates(totLength);
    int ind = 0;

    for (unsigned int i_time = 0; i_time < Ntime; i_time++) {
        for (unsigned int i_analog = 0; i_analog < Nanalogs; i_analog++) {
            ind = i_analog;
            ind += i_time * Nanalogs;
            analogsCriteria[ind] = m_analogsCriteria(i_time, i_analog);
            analogsDates[ind] = m_analogsDates(i_time, i_analog);
        }
    }

    // Write data
    ncFile.PutVarArray("target_dates", start1D, count1D, &m_targetDates(0));
    ncFile.PutVarArray("analog_criteria", start2D, count2D, &analogsCriteria[0]);
    ncFile.PutVarArray("analog_dates", start2D, count2D, &analogsDates[0]);

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();
    return true;
}

bool asResultsAnalogsDates::Load(const wxString &AlternateFilePath)
{
    // If we don't want to load, skip
    if (!m_loadIntermediateResults) {
        return false;
    } else {
        // Check if the current step is concerned
        ThreadsManager().CritSectionConfig().Enter();
        wxConfigBase *pConfig = wxFileConfig::Get();
        bool loadAnalogDatesStep1;
        pConfig->Read("/Optimizer/IntermediateResults/LoadAnalogDatesStep1", &loadAnalogDatesStep1, false);
        bool loadAnalogDatesStep2;
        pConfig->Read("/Optimizer/IntermediateResults/LoadAnalogDatesStep2", &loadAnalogDatesStep2, false);
        bool loadAnalogDatesStep3;
        pConfig->Read("/Optimizer/IntermediateResults/LoadAnalogDatesStep3", &loadAnalogDatesStep3, false);
        bool loadAnalogDatesStep4;
        pConfig->Read("/Optimizer/IntermediateResults/LoadAnalogDatesStep4", &loadAnalogDatesStep4, false);
        bool loadAnalogDatesAllSteps;
        pConfig->Read("/Optimizer/IntermediateResults/LoadAnalogDatesAllSteps", &loadAnalogDatesAllSteps, false);
        ThreadsManager().CritSectionConfig().Leave();

        if (!loadAnalogDatesAllSteps) {
            switch (m_currentStep) {
                case 0: {
                    if (!loadAnalogDatesStep1) {
                        if (!loadAnalogDatesStep2) {
                            if (!loadAnalogDatesStep3) {
                                if (!loadAnalogDatesStep4) {
                                    return false;
                                } else {
                                    m_currentStep = 3;
                                    BuildFileName();
                                }
                            } else {
                                m_currentStep = 2;
                                BuildFileName();
                            }
                        } else {
                            m_currentStep = 1;
                            BuildFileName();
                        }
                    }
                    break;
                }
                case 1: {
                    if (!loadAnalogDatesStep2) {
                        if (!loadAnalogDatesStep3) {
                            if (!loadAnalogDatesStep4) {
                                return false;
                            } else {
                                m_currentStep = 3;
                                BuildFileName();
                            }
                        } else {
                            m_currentStep = 2;
                            BuildFileName();
                        }
                    }
                    break;
                }
                case 2: {
                    if (!loadAnalogDatesStep3) {
                        if (!loadAnalogDatesStep4) {
                            return false;
                        } else {
                            m_currentStep = 3;
                            BuildFileName();
                        }
                    }
                    break;
                }
                case 3: {
                    if (!loadAnalogDatesStep4)
                        return false;
                    break;
                }
                default: {
                    return false;
                }
            }
        }
    }

    if (!Exists())
        return false;

    // Get the file path
    wxString ResultsFile;
    if (AlternateFilePath.IsEmpty()) {
        ResultsFile = m_filePath;
    } else {
        ResultsFile = AlternateFilePath;
    }

    ThreadsManager().CritSectionNetCDF().Enter();

    // Open the NetCDF file
    asFileNetcdf ncFile(ResultsFile, asFileNetcdf::ReadOnly);
    if (!ncFile.Open()) {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Get the elements size
    int Ntime = ncFile.GetDimLength("time");
    int Nanalogs = ncFile.GetDimLength("analogs");

    // Get time
    m_targetDates.resize(Ntime);
    ncFile.GetVar("target_dates", &m_targetDates[0]);

    // Check last value
    if (m_targetDates[m_targetDates.size() - 1] < m_targetDates[0]) {
        asLogError(_("The target date array is not consistent in the temp file (last value makes no sense)."));
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Create vectors for matrices data
    int totLength = Ntime * Nanalogs;
    VectorFloat analogsCriteria(totLength);
    VectorFloat analogsDates(totLength);

    // Get data
    size_t IndexStart[2] = {0, 0};
    size_t IndexCount[2] = {size_t(Ntime), size_t(Nanalogs)};
    ncFile.GetVarArray("analog_dates", IndexStart, IndexCount, &analogsDates[0]);
    ncFile.GetVarArray("analog_criteria", IndexStart, IndexCount, &analogsCriteria[0]);

    // Set data into the matrices
    m_analogsDates.resize(Ntime, Nanalogs);
    m_analogsCriteria.resize(Ntime, Nanalogs);
    int ind = 0;
    for (int i_time = 0; i_time < Ntime; i_time++) {
        for (int i_analog = 0; i_analog < Nanalogs; i_analog++) {
            ind = i_analog;
            ind += i_time * Nanalogs;
            m_analogsCriteria(i_time, i_analog) = analogsCriteria[ind];
            m_analogsDates(i_time, i_analog) = analogsDates[ind];
        }
    }

    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();
    return true;
}
