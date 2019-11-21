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

#ifndef AS_METHOD_STANDARD_H
#define AS_METHOD_STANDARD_H

#include <asIncludes.h>
#include <asPredictand.h>

class asPredictor;
class asPredictor;
class asParameters;
class asCriteria;

class asMethodStandard : public wxObject {
   public:
    asMethodStandard();

    ~asMethodStandard() override;

    virtual bool Manager();

    bool LoadPredictandDB(const wxString &predictandDBFilePath);

    void Cancel();

    bool PreloadArchiveData(asParameters *params);

    bool ProceedToArchiveDataPreloading(asParameters *params);

    bool CheckArchiveDataIsPreloaded(const asParameters *params) const;

    bool HasPreloadedArchiveData(int iStep, int iPtor) const;

    bool HasPreloadedArchiveData(int iStep, int iPtor, int iPre) const;

    bool PointersArchiveDataShared(asParameters *params, int iStep, int iPtor, int iPre);

    bool PreloadArchiveDataWithoutPreprocessing(asParameters *params, int iStep, int iPtor, int iDat);

    bool PreloadArchiveDataWithPreprocessing(asParameters *params, int iStep, int iPtor);

    bool LoadArchiveData(std::vector<asPredictor *> &predictors, asParameters *params, int iStep, double timeStartData,
                         double timeEndData);

    bool ExtractPreloadedArchiveData(std::vector<asPredictor *> &predictors, asParameters *params, int iStep,
                                     int iPtor);

    bool ExtractArchiveData(std::vector<asPredictor *> &predictors, asParameters *params, int iStep, int iPtor,
                            double timeStartData, double timeEndData);

    bool PreprocessArchiveData(std::vector<asPredictor *> &predictors, asParameters *params, int iStep, int iPtor,
                               double timeStartData, double timeEndData);

    bool GetRandomLevelValidData(asParameters *params, int iStep, int iPtor, int iPre, int iHour);

    bool GetRandomValidData(asParameters *params, int iStep, int iPtor, int iPre);

    void SetParamsFilePath(const wxString &val) {
        m_paramsFilePath = val;
    }

    void SetPredictandDBFilePath(const wxString &val) {
        m_predictandDBFilePath = val;
    }

    void SetPredictandDB(asPredictand *pDB) {
        m_predictandDB = pDB;
    }

    void SetPredictorDataDir(const wxString &val) {
        m_predictorDataDir = val;
    }

    bool IsArchiveDataPointerCopy(int iStep, int iPtor, int iPre) const {
        return m_preloadedArchivePointerCopy[iStep][iPtor][iPre];
    }

   protected:
    bool m_cancel;
    bool m_preloaded;
    bool m_warnFailedLoadingData;
    bool m_dumpPredictorData;
    bool m_loadFromDumpedData;
    wxString m_paramsFilePath;
    wxString m_predictandDBFilePath;
    wxString m_predictorDataDir;
    asPredictand *m_predictandDB;
    std::vector<std::vector<std::vector<std::vector<std::vector<asPredictor *> > > > > m_preloadedArchive;
    std::vector<vvb> m_preloadedArchivePointerCopy;

    bool Preprocess(std::vector<asPredictor *> predictors, const wxString &method, asPredictor *result);

    double GetTimeStartArchive(asParameters *params) const;

    double GetTimeEndArchive(asParameters *params) const;

    virtual void InitializePreloadedArchiveDataContainers(asParameters *params);

    virtual void Cleanup(std::vector<asPredictor *> predictors);

    virtual void Cleanup(std::vector<asCriteria *> criteria);

    void DeletePreloadedArchiveData();

    virtual double GetEffectiveArchiveDataStart(asParameters *params) const = 0;

    virtual double GetEffectiveArchiveDataEnd(asParameters *params) const = 0;

   private:
};

#endif
