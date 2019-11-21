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

#include "asIncludes.h"
#include "asPredictor.h"

class asPredictorOper : public asPredictor {
 public:
  explicit asPredictorOper(const wxString &dataId);

  ~asPredictorOper() override = default;

  static void SetDefaultPredictorsUrls();

  static asPredictorOper *GetInstance(const wxString &datasetId, const wxString &dataId);

  int Download();

  double UpdateRunDateInUse();

  double SetRunDateInUse(double val = 0);

  double DecrementRunDateInUse();

  bool BuildFilenamesUrls();

  void RestrictTimeArray(double restrictHours, double restrictTimeStepHours, int leadTimeNb);

  double GetForecastLeadTimeEnd() const {
    return m_leadTimeEnd;
  }

  double GetRunDateInUse() const {
    return m_runDateInUse;
  }

  vwxs GetUrls() const {
    return m_urls;
  }

  vwxs GetFileNames() const {
    return m_fileNames;
  }

  void SetFileNames(const vwxs &val) {
    m_fileNames = val;
  }

  vd GetDataDates() const {
    return m_dataDates;
  }

  void SetPredictorsRealtimeDirectory(const wxString &dir) {
    m_predictorsRealtimeDir = dir;
  }

 protected:
  wxString m_predictorsRealtimeDir;
  int m_leadTimeStart;
  int m_leadTimeEnd;
  int m_leadTimeStep;
  int m_runHourStart;
  int m_runUpdate;
  double m_runDateInUse;
  wxString m_commandDownload;
  bool m_restrictDownloads;
  int m_restrictHours;
  int m_restrictTimeStepHours;
  vwxs m_fileNames;
  vwxs m_urls;
  vd m_dataDates;

  void ListFiles(asTimeArray &timeArray) override;

  bool CheckTimeArray(asTimeArray &timeArray) override;
};

#endif
