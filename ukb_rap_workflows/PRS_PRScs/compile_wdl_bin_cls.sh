# Compile WDL for DNANexus
WOMTOOL_JAR=../dx_compile/womtool-86.jar
DX_COMPILER_JAR=../dx_compile/dxCompiler-2.11.4.jar
PROJID=project-GG25fB8Jv7B928vqK7k6vYY6		# UKB_Test

# Validate WDL
echo "Validating WDL"
java -jar "$WOMTOOL_JAR" validate prs_prscs_bin_cls.wdl

# Compile WDL
echo "Compiling WDL"
java -jar "$DX_COMPILER_JAR" compile prs_prscs_bin_cls.wdl \
	-project $PROJID \
	-folder /rdevito/nonlin_prs/ \
	-archive 