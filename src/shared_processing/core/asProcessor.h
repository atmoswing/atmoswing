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

#ifndef AS_PROCESSOR_H
#define AS_PROCESSOR_H

#include "asHeadersBase.h"

class asTimeArray;

class asParameters;

class asResultsDates;

class asCriteria;

class asResultsValues;

class asPredictor;

class asPredictand;

struct CudaCallbackParams;

// Was previously `class asProcessor : public wxObject` with all-static methods and no instance
// state. Converted to a namespace — the call-site syntax `asProcessor::Foo(...)` works identically
// for both. Verified: no derivations, no instantiations.
// Internal helpers (CheckArchiveTimeArray, CheckTargetTimeArray) live in an anonymous
// namespace inside asProcessor.cpp rather than being declared here.
namespace asProcessor {

/**
 * Predictor data prepared for the analog scan: a pointer table indexed by
 * [iTime * membersNb + iMem] giving direct access to each contiguous grid,
 * optionally backed by a flattened copy of the whole dataset for better cache
 * locality (candidates close in time then sit close in memory).
 */
struct FlatPredictorData {
    vf storage;                      // owns the flattened copy (empty when not copying)
    std::vector<const float*> ptrs;  // [iTime * membersNb + iMem] -> grid data
    int membersNb = 1;
};

std::vector<FlatPredictorData> FlattenPredictors(const vector<asPredictor*>& predictors, bool copyData);

bool GetAnalogsDates(vector<asPredictor*> predictorsArchive, vector<asPredictor*> predictorsTarget,
                     asTimeArray& timeArrayArchiveData, asTimeArray& timeArrayArchiveSelection,
                     asTimeArray& timeArrayTargetData, asTimeArray& timeArrayTargetSelection,
                     vector<asCriteria*> criteria, asParameters* params, int step, asResultsDates& results,
                     bool& containsNaNs);

bool GetAnalogsSubDates(vector<asPredictor*> predictorsArchive, vector<asPredictor*> predictorsTarget,
                        asTimeArray& timeArrayArchiveData, asTimeArray& timeArrayTargetData, asResultsDates& anaDates,
                        vector<asCriteria*> criteria, asParameters* params, int step, asResultsDates& results,
                        bool& containsNaNs);

bool GetAnalogsValues(asPredictand& predictand, asResultsDates& anaDates, asParameters* params,
                      asResultsValues& results);

void InsertInArrays(bool isAsc, int analogsNb, float analogDate, float score, int counter, a1f& scoreArrayOneDay,
                    a1f& dateArrayOneDay);

void InsertInArraysNoDuplicate(bool isAsc, int analogsNb, float analogDate, float score, a1f& scoreArrayOneDay,
                               a1f& dateArrayOneDay);

int FindNextDate(asTimeArray& dateArray, a1d& timeData, int iTimeStart, int iDate);

int FindNextDate(a1d& dateArray, a1d& timeData, int iTimeStart, int iDate);

}  // namespace asProcessor

#endif
