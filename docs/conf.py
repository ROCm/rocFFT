from rocm_docs import ROCmDocs

docs_core = ROCmDocs("rocFFT Documentation")
docs_core.run_doxygen()
docs_core.setup()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
