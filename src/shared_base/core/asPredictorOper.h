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

#ifndef AS_PREDICTOR_OPER_H
#define AS_PREDICTOR_OPER_H

#include "asPredictor.h"

class asPredictorOper : public asPredictor {
  public:
    explicit asPredictorOper(const wxString& dataId);

    ~asPredictorOper() override = default;

    static void SetDefaultPredictorsUrls();

    static asPredictorOper* GetInstance(const wxString& datasetId, const wxString& dataId);

    int Download();

    double UpdateRunDateInUse();

    double SetRunDateInUse(double val = 0);

    double DecrementRunDateInUse();

    bool BuildFilenamesAndUrls(double predictorHour, double forecastTimeStepHours, int leadTimeNb);

    double GetRunDateInUse() const {
        return _runDateInUse;
    }

    vwxs GetUrls() const {
        return _urls;
    }

    vwxs GetFileNames() const {
        return _fileNames;
    }

    void SetFileNames(const vwxs& val) {
        _fileNames = val;
    }

    vd GetDataDates() const {
        return _dataDates;
    }

    wxString GetPredictorsRealtimeDirectory() {
        return _predictorsRealtimeDir;
    }

    void SetPredictorsRealtimeDirectory(const wxString& dir) {
        _predictorsRealtimeDir = dir;
    }

    bool ShouldDownload() {
        return _shouldDownload;
    }

    virtual wxString GetDirStructure(const double date);

    virtual wxString GetFileName(const double date, const int leadTime);

  protected:
    wxString _predictorsRealtimeDir;
    int _leadTimeStart;
    int _leadTimeStep;
    int _runHourStart;
    int _runUpdate;
    double _runDateInUse;
    wxString _commandDownload;
    bool _shouldDownload;
    vwxs _fileNames;
    vwxs _urls;
    vd _dataDates;

    void ListFiles(asTimeArray& timeArray) override;

    bool ExtractFromFiles(asAreaGrid*& dataArea, asTimeArray& timeArray) override;

    bool CheckTimeArray(asTimeArray& timeArray) override;
};

#endif
