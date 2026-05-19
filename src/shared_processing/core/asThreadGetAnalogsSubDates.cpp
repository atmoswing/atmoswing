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

#include "asThreadGetAnalogsSubDates.h"

#include <utility>

#include "asCriteria.h"
#include "asPredictor.h"
#include "asTimeArray.h"

asThreadGetAnalogsSubDates::asThreadGetAnalogsSubDates(
    vector<asPredictor*> predictorsArchive, vector<asPredictor*> predictorsTarget, asTimeArray* timeArrayArchiveData,
    asTimeArray* timeArrayTargetData, a1f* timeTargetSelection, vector<asCriteria*> criteria, asParameters* params,
    int step, vpa2f& vTargData, vpa2f& vArchData, a1i& vRowsNb, a1i& vColsNb, int start, int end,
    a2f* finalAnalogsCriteria, a2f* finalAnalogsDates, a2f* previousAnalogsDates, bool* containsNaNs, bool* success)
    : asThread(asThread::ProcessorGetAnalogsDates),
      _pPredictorsArchive(std::move(predictorsArchive)),
      _pPredictorsTarget(std::move(predictorsTarget)),
      _pTimeArrayArchiveData(timeArrayArchiveData),
      _pTimeArrayTargetData(timeArrayTargetData),
      _pTimeTargetSelection(timeTargetSelection),
      _criteria(std::move(criteria)),
      _params(params),
      _vTargData(vTargData),
      _vArchData(vArchData),
      _vRowsNb(vRowsNb),
      _vColsNb(vColsNb),
      _pFinalAnalogsCriteria(finalAnalogsCriteria),
      _pFinalAnalogsDates(finalAnalogsDates),
      _pPreviousAnalogsDates(previousAnalogsDates),
      _success(success) {
    _step = step;
    _start = start;
    _end = end;
    _pContainsNaNs = containsNaNs;

    wxASSERT_MSG(_end < _pTimeTargetSelection->size(),
                 _("The given time array end is superior to the time array size."));
    wxASSERT_MSG(_end != _pTimeTargetSelection->size() - 2,
                 asStrF(_("The given time array end is missing its last value (end=%d, size=%d)."), _end,
                        (int)_pTimeTargetSelection->size()));
}

asThreadGetAnalogsSubDates::~asThreadGetAnalogsSubDates() {}

wxThread::ExitCode asThreadGetAnalogsSubDates::Entry() {
    // Extract time arrays
    a1d timeArchiveData = _pTimeArrayArchiveData->GetTimeArray();
    a1d timeTargetData = _pTimeArrayTargetData->GetTimeArray();

    // Some other variables
    float tmpscore, thisscore;
    int timeArchiveDataSize = timeArchiveData.size();
    int timeTargetDataSize = timeTargetData.size();
    int predictorsNb = _params->GetPredictorsNb(_step);
    int membersNb = (_pPredictorsTarget)[0]->GetData()[0].size();
    int analogsNbPrevious = _params->GetAnalogsNumber(_step - 1);
    int analogsNb = _params->GetAnalogsNumber(_step);
    bool isasc = (_criteria[0]->GetOrder() == Asc);

    wxASSERT(_end < _pTimeTargetSelection->size());
    wxASSERT(timeArchiveDataSize == (int)(_pPredictorsArchive)[0]->GetData().size());
    wxASSERT(timeTargetDataSize == (int)(_pPredictorsTarget)[0]->GetData().size());
    wxASSERT(membersNb == (_pPredictorsArchive)[0]->GetData()[0].size());

    // Containers for daily results
    a1f currentAnalogsDates(analogsNbPrevious);
    a1f scoreArrayOneDay(analogsNb);
    scoreArrayOneDay.fill(NAN);
    a1f dateArrayOneDay(analogsNb);
    dateArrayOneDay.fill(NAN);

    // Loop through every timestep as target data
    // Former, but disabled: for (int iDateTarg=_start; !ThreadsManager().Cancelled() && (iDateTarg<=_end);
    // iDateTarg++)
    for (int iDateTarg = _start; iDateTarg <= _end; iDateTarg++) {
        int iTimeTarg = asFind(&timeTargetData[0], &timeTargetData[timeTargetDataSize - 1],
                               (double)_pTimeTargetSelection->coeff(iDateTarg), 0.01);
        wxASSERT(_pTimeTargetSelection->coeff(iDateTarg) > 0);
        wxASSERT(iTimeTarg >= 0);
        if (iTimeTarg < 0) {
            wxLogError(_("An unexpected error occurred."));
            *_success = false;
            return (wxThread::ExitCode)-1;
        }

        // Get dates
        currentAnalogsDates = _pPreviousAnalogsDates->row(iDateTarg);

        // Counter representing the current index
        int counter = 0;

        scoreArrayOneDay.fill(NAN);
        dateArrayOneDay.fill(NAN);

        // Loop over the members
        for (int iMem = 0; iMem < membersNb; ++iMem) {
            // Extract target data
            for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                _vTargData[iPtor] = &(_pPredictorsTarget)[iPtor]->GetData()[iTimeTarg][iMem];
            }

            // Loop through the previous analogs for candidate data
            for (int iPrevAnalog = 0; iPrevAnalog < analogsNbPrevious; iPrevAnalog++) {
                if (isnan(currentAnalogsDates[iPrevAnalog])) {
                    *_pContainsNaNs = true;
                    continue;
                }

                // Find row in the predictor time array
                int iTimeArch = asFind(&timeArchiveData[0], &timeArchiveData[timeArchiveDataSize - 1],
                                       currentAnalogsDates[iPrevAnalog], 0.01);
                wxASSERT(iTimeArch >= 0);
                if (iTimeArch < 0) {
                    wxLogError(_("An unexpected error occurred."));
                    *_success = false;
                    return (wxThread::ExitCode)-1;
                }

                // Check if a row was found
                if (iTimeArch != asNOT_FOUND && iTimeArch != asOUT_OF_RANGE) {
                    // Process the criteria
                    thisscore = 0;
                    for (int iPtor = 0; iPtor < predictorsNb; iPtor++) {
                        // Get data
                        _vArchData[iPtor] = &(_pPredictorsArchive)[iPtor]->GetData()[iTimeArch][iMem];

                        // Assess the criteria
                        wxASSERT(_criteria.size() > iPtor);
                        tmpscore = _criteria[iPtor]->Assess(*_vTargData[iPtor], *_vArchData[iPtor], _vRowsNb[iPtor],
                                                             _vColsNb[iPtor]);

                        // Weight and add the score
                        thisscore += tmpscore * _params->GetPredictorWeight(_step, iPtor);
                    }
                    if (isnan(thisscore)) {
                        *_pContainsNaNs = true;
                        continue;
                    }

                    // Check if the array is already full
                    if (counter > analogsNb - 1) {
                        if (isasc) {
                            if (thisscore < scoreArrayOneDay[analogsNb - 1]) {
                                asArraysInsert(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1],
                                               &dateArrayOneDay[0], &dateArrayOneDay[analogsNb - 1], Asc, thisscore,
                                               (float)timeArchiveData[iTimeArch]);
                            }
                        } else {
                            if (thisscore > scoreArrayOneDay[analogsNb - 1]) {
                                asArraysInsert(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1],
                                               &dateArrayOneDay[0], &dateArrayOneDay[analogsNb - 1], Desc, thisscore,
                                               (float)timeArchiveData[iTimeArch]);
                            }
                        }
                    } else if (counter < analogsNb - 1) {
                        // Add score and date to the vectors
                        scoreArrayOneDay[counter] = thisscore;
                        dateArrayOneDay[counter] = (float)timeArchiveData[iTimeArch];
                    } else {
                        // Add score and date to the vectors
                        scoreArrayOneDay[counter] = thisscore;
                        dateArrayOneDay[counter] = (float)timeArchiveData[iTimeArch];

                        // Sort both scores and dates arrays
                        if (isasc) {
                            asSortArrays(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1], &dateArrayOneDay[0],
                                         &dateArrayOneDay[analogsNb - 1], Asc);
                        } else {
                            asSortArrays(&scoreArrayOneDay[0], &scoreArrayOneDay[analogsNb - 1], &dateArrayOneDay[0],
                                         &dateArrayOneDay[analogsNb - 1], Desc);
                        }
                    }

                    counter++;
                } else {
                    wxLogError(
                        _("The date was not found in the array (Analogs subdates fct). "
                          "That should not happen."));
                    *_success = false;
                    return (wxThread::ExitCode)-1;
                }
            }
        }

        // Copy results
        _pFinalAnalogsCriteria->row(iDateTarg) = scoreArrayOneDay.head(analogsNb).transpose();
        _pFinalAnalogsDates->row(iDateTarg) = dateArrayOneDay.head(analogsNb).transpose();
    }

    *_success = true;

    return (wxThread::ExitCode)0;
}
