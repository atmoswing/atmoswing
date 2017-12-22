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
 * Portions Copyright 2017 Pascal Horton, University of Bern.
 */

#include "asMethodDownscaler.h"
#include <asIncludes.h>
#include <asParametersDownscaling.h>
#include <asResultsDates.h>
#include <asPredictorScenario.h>
#include <asResultsDates.h>
#include <asResultsValues.h>
#include <asCriteria.h>
#include <asGeoAreaCompositeGrid.h>
#include <asTimeArray.h>
#include <asProcessor.h>
#include <asPreprocessor.h>

#ifndef UNIT_TESTING

#endif

asMethodDownscaler::asMethodDownscaler()
        : asMethodStandard()
{
    // Seeds the random generator
    asTools::InitRandom();
}

asMethodDownscaler::~asMethodDownscaler()
{
    DeletePreloadedArchiveData();
}

bool asMethodDownscaler::Manager()
{
    // Set unresponsive to speedup
    g_responsive = false;

    // Seeds the random generator
    asTools::InitRandom();

    // Load parameters
    asParametersDownscaling params;
    if (!params.LoadFromFile(m_paramsFilePath)) {
        return false;
    }
    if (!m_predictandStationIds.empty()) {
        vvi idsVect;
        idsVect.push_back(m_predictandStationIds);
        params.SetPredictandStationIdsVector(idsVect);
    }
    params.InitValues();

    // Load the Predictand DB
    if (!LoadPredictandDB(m_predictandDBFilePath)) {
        return false;
    }

    // Watch
    wxStopWatch sw;

    // Downscale
    if (Downscale(params)) {
        // Display processing time
        wxLogMessage(_("The whole processing took %.3f min to execute"), float(sw.Time()) / 60000.0f);
#if wxUSE_GUI
        wxLogStatus(_("Downscaling over."));
#endif
    } else {
        wxLogError(_("The series could not be downscaled"));
    }

    // Delete preloaded data and cleanup
    DeletePreloadedArchiveData();

    return true;
}


void asMethodDownscaler::ClearAll()
{
    m_parameters.clear();
}

double asMethodDownscaler::GetTimeStartDownscaling(asParametersDownscaling *params) const
{
    double timeStartDownscaling = params->GetDownscalingStart();
    timeStartDownscaling += std::abs(params->GetTimeShiftDays());

    return timeStartDownscaling;
}

double asMethodDownscaler::GetTimeEndDownscaling(asParametersDownscaling *params) const
{
    double timeEndDownscaling = params->GetDownscalingEnd();
    timeEndDownscaling = wxMin(timeEndDownscaling, timeEndDownscaling - params->GetTimeSpanDays());

    return timeEndDownscaling;
}

double asMethodDownscaler::GetEffectiveArchiveDataStart(asParameters *params) const
{
    return GetTimeStartArchive(params);
}

double asMethodDownscaler::GetEffectiveArchiveDataEnd(asParameters *params) const
{
    return GetTimeEndArchive(params);
}

bool asMethodDownscaler::GetAnalogsDates(asResultsDates &results, asParametersDownscaling *params, int iStep,
                                         bool &containsNaNs)
{
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init(params);

    // Archive date array
    asTimeArray timeArrayArchive(GetTimeStartArchive(params), GetTimeEndArchive(params),
                                 params->GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
    if (!timeArrayArchive.Init()) {
        wxLogError(_("The time array mode for the archive dates is not correctly defined."));
        return false;
    }

    // Check on the archive length
    if (timeArrayArchive.GetSize() < 100) {
        wxLogError(_("The time array is not consistent in asMethodDownscaler::GetAnalogsDates: size=%d."),
                   timeArrayArchive.GetSize());
        return false;
    }

    // Target date array
    asTimeArray timeArrayTarget(GetTimeStartDownscaling(params), GetTimeEndDownscaling(params),
                                params->GetTimeArrayTargetTimeStepHours(), params->GetTimeArrayTargetMode());
    if (!timeArrayTarget.Init()) {
        wxLogError(_("The time array mode for the target dates is not correctly defined."));
        return false;
    }

    // Load the archive data
    std::vector<asPredictor *> predictorsArch;
    if (!LoadArchiveData(predictorsArch, params, iStep, GetTimeStartArchive(params), GetTimeEndArchive(params))) {
        wxLogError(_("Failed loading predictor data."));
        Cleanup(predictorsArch);
        return false;
    }

    // Load the scenario data
    std::vector<asPredictor *> predictorsScenario;
    if (!LoadScenarioulationData(predictorsScenario, params, iStep, GetTimeStartArchive(params), GetTimeEndArchive(params))) {
        wxLogError(_("Failed loading predictor data."));
        Cleanup(predictorsArch);
        Cleanup(predictorsScenario);
        return false;
    }

    // Create the criterion
    std::vector<asCriteria *> criteria;
    for (int iPtor = 0; iPtor < params->GetPredictorsNb(iStep); iPtor++) {
        // Instantiate a score object
        asCriteria *criterion = asCriteria::GetInstance(params->GetPredictorCriteria(iStep, iPtor));
        if (criterion->NeedsDataRange()) {
            wxASSERT(predictorsArch.size() > iPtor);
            wxASSERT(predictorsArch[iPtor]);
            criterion->SetDataRange(predictorsArch[iPtor]);
        }
        criteria.push_back(criterion);
    }

    // Check time sizes
#ifdef _DEBUG
    int prevTimeSize = 0;

    for (unsigned int i = 0; i < predictors.size(); i++) {
        if (i > 0) {
            wxASSERT(predictors[i]->GetTimeSize() == prevTimeSize);
        }
        prevTimeSize = predictors[i]->GetTimeSize();
    }
#endif // _DEBUG

    // Inline the data when possible
    for (int iPtor = 0; iPtor < (int) predictorsArch.size(); iPtor++) {
        if (criteria[iPtor]->CanUseInline()) {
            predictorsArch[iPtor]->Inline();
        }
    }

    // Send data and criteria to processor
    wxLogVerbose(_("Start processing the comparison."));

    if (!asProcessor::GetAnalogsDates(predictorsArch, predictorsScenario, timeArrayArchive, timeArrayArchive, timeArrayTarget,
                                      timeArrayTarget, criteria, params, iStep, results, containsNaNs)) {
        wxLogError(_("Failed processing the analogs dates."));
        Cleanup(predictorsArch);
        Cleanup(predictorsScenario);
        Cleanup(criteria);
        return false;
    }
    wxLogVerbose(_("The processing is over."));

    Cleanup(predictorsArch);
    Cleanup(predictorsScenario);
    Cleanup(criteria);

    return true;
}

bool asMethodDownscaler::GetAnalogsSubDates(asResultsDates &results, asParametersDownscaling *params,
                                            asResultsDates &anaDates, int iStep, bool &containsNaNs)
{
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init(params);

    // Date array object instantiation for the processor
    wxLogVerbose(_("Creating a date arrays for the processor."));
    double timeStart = params->GetArchiveStart();
    double timeEnd = params->GetArchiveEnd();
    timeEnd = wxMin(timeEnd, timeEnd - params->GetTimeSpanDays()); // Adjust so the predictors search won't overtake the array
    asTimeArray timeArrayArchive(timeStart, timeEnd, params->GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
    timeArrayArchive.Init();
    wxLogVerbose(_("Date arrays created."));

    // Load the predictor data
    std::vector<asPredictor *> predictors;
    if (!LoadArchiveData(predictors, params, iStep, timeStart, timeEnd)) {
        wxLogError(_("Failed loading predictor data."));
        Cleanup(predictors);
        return false;
    }

    // Create the score objects
    std::vector<asCriteria *> criteria;
    for (int iPtor = 0; iPtor < params->GetPredictorsNb(iStep); iPtor++) {
        wxLogVerbose(_("Creating a criterion object."));
        asCriteria *criterion = asCriteria::GetInstance(params->GetPredictorCriteria(iStep, iPtor));
        if (criterion->NeedsDataRange()) {
            wxASSERT(predictors.size() > iPtor);
            wxASSERT(predictors[iPtor]);
            criterion->SetDataRange(predictors[iPtor]);
        }
        criteria.push_back(criterion);
        wxLogVerbose(_("Criterion object created."));
    }

    // Inline the data when possible
    for (int iPtor = 0; iPtor < (int) predictors.size(); iPtor++) {
        if (criteria[iPtor]->CanUseInline()) {
            predictors[iPtor]->Inline();
        }
    }

    // Send data and criteria to processor
    wxLogVerbose(_("Start processing the comparison."));
    if (!asProcessor::GetAnalogsSubDates(predictors, predictors, timeArrayArchive, timeArrayArchive, anaDates, criteria,
                                         params, iStep, results, containsNaNs)) {
        wxLogError(_("Failed processing the analogs dates."));
        Cleanup(predictors);
        Cleanup(criteria);
        return false;
    }
    wxLogVerbose(_("The processing is over."));

    Cleanup(predictors);
    Cleanup(criteria);

    return true;
}

bool asMethodDownscaler::GetAnalogsValues(asResultsValues &results, asParametersDownscaling *params,
                                          asResultsDates &anaDates, int iStep)
{
    // Initialize the result object
    results.SetCurrentStep(iStep);
    results.Init(params);

    // Set the predictand values to the corresponding analog dates
    wxASSERT(m_predictandDB);
    wxLogVerbose(_("Start setting the predictand values to the corresponding analog dates."));
    if (!asProcessor::GetAnalogsValues(*m_predictandDB, anaDates, params, results)) {
        wxLogError(_("Failed setting the predictand values to the corresponding analog dates."));
        return false;
    }
    wxLogVerbose(_("Predictand association over."));

    return true;
}

bool asMethodDownscaler::SaveDetails(asParametersDownscaling *params)
{
    asResultsDates anaDatesPrevious;
    asResultsDates anaDates;
    asResultsValues anaValues;

    // Process every step one after the other
    int stepsNb = params->GetStepsNb();
    for (int iStep = 0; iStep < stepsNb; iStep++) {
        bool containsNaNs = false;
        if (iStep == 0) {
            if (!GetAnalogsDates(anaDates, params, iStep, containsNaNs))
                return false;
        } else {
            anaDatesPrevious = anaDates;
            if (!GetAnalogsSubDates(anaDates, params, anaDatesPrevious, iStep, containsNaNs))
                return false;
        }
        if (containsNaNs) {
            wxLogError(_("The dates selection contains NaNs"));
            return false;
        }
    }
    if (!GetAnalogsValues(anaValues, params, anaDates, stepsNb - 1))
        return false;

    anaDates.SetSubFolder("downscaling");
    anaDates.Save();
    anaValues.SetSubFolder("downscaling");
    anaValues.Save();

    return true;
}

bool asMethodDownscaler::LoadScenarioulationData(std::vector<asPredictor *> &predictors, asParametersDownscaling *params,
                                                 int iStep, double timeStartData, double timeEndData)
{
    try {
        // Loop through every predictor
        for (int iPtor = 0; iPtor < params->GetPredictorsNb(iStep); iPtor++) {
            wxLogVerbose(_("Loading model scenario."));

            if (!params->NeedsPreprocessing(iStep, iPtor)) {
                if (!ExtractScenarioulationDataWithoutPreprocessing(predictors, params, iStep, iPtor, timeStartData,
                                                                    timeEndData)) {
                    return false;
                }
            } else {
                if (!ExtractScenarioulationDataWithPreprocessing(predictors, params, iStep, iPtor, timeStartData,
                                                                 timeEndData)) {
                    return false;
                }
            }

            wxLogVerbose(_("Scenario data loaded."));
        }
    } catch (std::bad_alloc &ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation in the scenario data loading: %s"), msg);
        return false;
    } catch (asException &e) {
        wxString fullMessage = e.GetFullMessage();
        if (!fullMessage.IsEmpty()) {
            wxLogError(fullMessage);
        }
        wxLogError(_("Failed to load data."));
        return false;
    } catch (std::exception &e) {
        wxString msg(e.what(), wxConvUTF8);
        wxLogError(_("Exception in the scenario data loading: %s"), msg);
        return false;
    }

    return true;
}

bool asMethodDownscaler::ExtractScenarioulationDataWithoutPreprocessing(std::vector<asPredictor *> &predictors,
                                                                        asParametersDownscaling *params, int iStep,
                                                                        int iPtor, double timeStartData,
                                                                        double timeEndData)
{
    // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive,
    // and the predictor dates are aligned with the target dates, but the dates are not the same.
    double ptorStart = timeStartData - params->GetTimeShiftDays() + params->GetPredictorTimeHours(iStep, iPtor) / 24.0;
    double ptorEnd = timeEndData - params->GetTimeShiftDays() + params->GetPredictorTimeHours(iStep, iPtor) / 24.0;
    asTimeArray timeArray(ptorStart, ptorEnd, params->GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
    timeArray.Init();

    // Loading the datasets information
    asPredictorScenario *predictor = asPredictorScenario::GetInstance(params->GetPredictorScenarioDatasetId(iStep, iPtor),
                                                                  params->GetPredictorScenarioDataId(iStep, iPtor),
                                                                  m_predictorScenarioDataDir);
    if (!predictor) {
        return false;
    }

    // Select the number of members for ensemble data.
    if (predictor->IsEnsemble()) {
        predictor->SelectMembers(params->GetPredictorScenarioMembersNb(iStep, iPtor));
    }

    // Area object instantiation
    asGeoAreaCompositeGrid *area = asGeoAreaCompositeGrid::GetInstance(
            params->GetPredictorGridType(iStep, iPtor),
            params->GetPredictorXmin(iStep, iPtor),
            params->GetPredictorXptsnb(iStep, iPtor),
            params->GetPredictorXstep(iStep, iPtor),
            params->GetPredictorYmin(iStep, iPtor),
            params->GetPredictorYptsnb(iStep, iPtor),
            params->GetPredictorYstep(iStep, iPtor),
            params->GetPredictorLevel(iStep, iPtor), asNONE,
            params->GetPredictorFlatAllowed(iStep, iPtor));
    wxASSERT(area);

    // Check the starting dates coherence
    if (predictor->GetOriginalProviderStart() > ptorStart) {
        wxLogError(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodDownscaler::GetAnalogsDates, no preprocessing)."),
                   asTime::GetStringTime(ptorStart), asTime::GetStringTime(predictor->GetOriginalProviderStart()));
        wxDELETE(area);
        wxDELETE(predictor);
        return false;
    }

    // Data loading
    if (!predictor->Load(area, timeArray)) {
        wxLogError(_("The data could not be loaded."));
        wxDELETE(area);
        wxDELETE(predictor);
        return false;
    }
    wxDELETE(area);
    predictors.push_back(predictor);

    return true;
}

bool asMethodDownscaler::ExtractScenarioulationDataWithPreprocessing(std::vector<asPredictor *> &predictors,
                                                                     asParametersDownscaling *params, int iStep,
                                                                     int iPtor, double timeStartData,
                                                                     double timeEndData)
{
    std::vector<asPredictorScenario *> predictorsPreprocess;

    int preprocessSize = params->GetPreprocessSize(iStep, iPtor);

    wxLogVerbose(_("Preprocessing data (%d predictor(s)) while loading."), preprocessSize);

    for (int iPre = 0; iPre < preprocessSize; iPre++) {
        // Date array object instantiation for the data loading. The array has the same length than timeArrayArchive,
        // and the predictor dates are aligned with the target dates, but the dates are not the same.
        double ptorStart = timeStartData - double(params->GetTimeShiftDays()) +
                           params->GetPreprocessTimeHours(iStep, iPtor, iPre) / 24.0;
        double ptorEnd = timeEndData - double(params->GetTimeShiftDays()) +
                         params->GetPreprocessTimeHours(iStep, iPtor, iPre) / 24.0;
        asTimeArray timeArray(ptorStart, ptorEnd, params->GetTimeArrayAnalogsTimeStepHours(), asTimeArray::Simple);
        timeArray.Init();

        // Loading the dataset information
        asPredictorScenario *predictorPreprocess = asPredictorScenario::GetInstance(
                params->GetPreprocessScenarioDatasetId(iStep, iPtor, iPre),
                params->GetPreprocessScenarioDataId(iStep, iPtor, iPre),
                m_predictorScenarioDataDir);
        if (!predictorPreprocess) {
            Cleanup(predictorsPreprocess);
            return false;
        }

        // Select the number of members for ensemble data.
        if (predictorPreprocess->IsEnsemble()) {
            predictorPreprocess->SelectMembers(params->GetPreprocessMembersNb(iStep, iPtor, iPre));
        }

        // Area object instantiation
        asGeoAreaCompositeGrid *area = asGeoAreaCompositeGrid::GetInstance(
                params->GetPredictorGridType(iStep, iPtor),
                params->GetPredictorXmin(iStep, iPtor),
                params->GetPredictorXptsnb(iStep, iPtor),
                params->GetPredictorXstep(iStep, iPtor),
                params->GetPredictorYmin(iStep, iPtor),
                params->GetPredictorYptsnb(iStep, iPtor),
                params->GetPredictorYstep(iStep, iPtor),
                params->GetPreprocessLevel(iStep, iPtor, iPre),
                asNONE,
                params->GetPredictorFlatAllowed(iStep, iPtor));
        wxASSERT(area);

        // Check the starting dates coherence
        if (predictorPreprocess->GetOriginalProviderStart() > ptorStart) {
            wxLogError(_("The first year defined in the parameters (%s) is prior to the start date of the data (%s) (in asMethodDownscaler::GetAnalogsDates, preprocessing)."),
                       asTime::GetStringTime(ptorStart),
                       asTime::GetStringTime(predictorPreprocess->GetOriginalProviderStart()));
            wxDELETE(area);
            wxDELETE(predictorPreprocess);
            Cleanup(predictorsPreprocess);
            return false;
        }

        // Data loading
        if (!predictorPreprocess->Load(area, timeArray)) {
            wxLogError(_("The data could not be loaded."));
            wxDELETE(area);
            wxDELETE(predictorPreprocess);
            Cleanup(predictorsPreprocess);
            return false;
        }
        wxDELETE(area);
        predictorsPreprocess.push_back(predictorPreprocess);
    }

    // Fix the criteria if S1
    if (params->GetPreprocessMethod(iStep, iPtor).IsSameAs("Gradients") &&
        params->GetPredictorCriteria(iStep, iPtor).IsSameAs("S1")) {
        params->SetPredictorCriteria(iStep, iPtor, "S1grads");
    } else if (params->GetPreprocessMethod(iStep, iPtor).IsSameAs("Gradients") &&
               params->GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1")) {
        params->SetPredictorCriteria(iStep, iPtor, "NS1grads");
    }

    auto *predictor = new asPredictorScenario(*predictorsPreprocess[0]);
    if (!Preprocess(predictorsPreprocess, params->GetPreprocessMethod(iStep, iPtor), predictor)) {
        wxLogError(_("Data preprocessing failed."));
        Cleanup(predictorsPreprocess);
        wxDELETE(predictor);
        return false;
    }

    Cleanup(predictorsPreprocess);
    predictors.push_back(predictor);

    return true;
}

bool asMethodDownscaler::Preprocess(std::vector<asPredictorScenario *> predictors, const wxString &method, asPredictor *result)
{
    std::vector<asPredictor *> ptorsPredictors(predictors.begin(), predictors.end());

    return asPreprocessor::Preprocess(ptorsPredictors, method, result);
}

void asMethodDownscaler::Cleanup(std::vector<asPredictorScenario *> predictors)
{
    if (!predictors.empty()) {
        for (auto &predictor : predictors) {
            wxDELETE(predictor);
        }
        predictors.resize(0);
    }
}

void asMethodDownscaler::Cleanup(std::vector<asPredictorArch *> predictors)
{
    if (!predictors.empty()) {
        for (auto &predictor : predictors) {
            wxDELETE(predictor);
        }
        predictors.resize(0);
    }
}

void asMethodDownscaler::Cleanup(std::vector<asPredictor *> predictors)
{
    if (!predictors.empty()) {
        for (auto &predictor : predictors) {
            wxDELETE(predictor);
        }
        predictors.resize(0);
    }
}

void asMethodDownscaler::Cleanup(std::vector<asCriteria *> criteria)
{
    if (!criteria.empty()) {
        for (unsigned int i = 0; i < criteria.size(); i++) {
            wxDELETE(criteria[i]);
        }
        criteria.resize(0);
    }
}